import math


class Trip:

    def __init__(self, name, distance_in_km, load_in_ton, slope_profile, empty_trip_factor, during_iteration):
        self.name = name
        self.distance_in_km = distance_in_km
        self.load_in_ton = load_in_ton
        self.slope_profile = slope_profile
        self.empty_trip_factor = empty_trip_factor
        self.emissions_per_mode = {}
        self.during_iteration = during_iteration
    def compute_train_wtw_emissions_in_tCO2e(self, setting):

        wtw_emissions_in_tCO2e = Train(setting=setting).compute_wtw_emissions_in_tCO2e(trip=self)

        return wtw_emissions_in_tCO2e

    def compute_truck_wtw_emissions_in_tCO2e(self, setting):

        wtw_emissions_in_tCO2e = Truck(setting=setting).compute_wtw_emissions_in_tCO2e(trip=self)

        return wtw_emissions_in_tCO2e

    def compute_barge_wtw_emissions_in_tCO2e(self, setting):
        wtw_emissions_in_tCO2e = Barge(setting=setting).compute_wtw_emissions_in_tCO2e(trip=self)

        return wtw_emissions_in_tCO2e


class Train:

    def __init__(self, setting):

        if setting == 'diesel' or setting == 'default':
            self.weight_locomotive_in_ton = 123
            self.weight_empty_wagon_in_ton = 23
            self.load_per_wagon_in_ton = 40         # this is the actual max. load per wagon
            self.max_load_per_wagon_in_ton = 61     # this is required to calc. the cap. utilization of wagons
            self.max_number_of_wagons_per_train = 32
            self.max_load_train_in_ton = self.load_per_wagon_in_ton * self.max_number_of_wagons_per_train

            # energy-emission factors
            self.engine_type = 'diesel'
            self.engine_diesel_electric_efficiency = 0.37
            self.ttw_energy_in_MJ_to_wtw_in_gCO2e = 88.21

        elif setting == 'electric':
            self.weight_locomotive_in_ton = 115
            self.weight_empty_wagon_in_ton = 23
            self.load_per_wagon_in_ton = 40         # this is the actual max. load per wagon
            self.max_load_per_wagon_in_ton = 60     # this is required to calc. the cap. utilization of wagons
            self.max_number_of_wagons_per_train = 32
            self.max_load_train_in_ton = self.load_per_wagon_in_ton * self.max_number_of_wagons_per_train

            # energy-emission factors
            self.engine_type = 'electric'
            self.engine_diesel_electric_efficiency = 0.37
            self.ttw_energy_in_MJ_to_wtw_in_gCO2e = 117 # EU-28: 117 gCO2e/MJ, Germany: 161 gCO2e/MJ (see Table 52, p. 109)

        else:
            print('train type not defined')

    def compute_wtw_emissions_in_tCO2e(self, trip):

        wtw_emissions_in_tCO2e = 0

        # this loop handles cases in which the trip encompasses multiple trains
        remaining_load_in_ton = trip.load_in_ton
        while remaining_load_in_ton > 0:
            train_load_in_ton = min(remaining_load_in_ton, self.max_load_train_in_ton)
            number_of_wagons = math.ceil(train_load_in_ton / self.load_per_wagon_in_ton)

            net_weight_in_ton = train_load_in_ton
            gross_weight_in_ton = self.weight_locomotive_in_ton + train_load_in_ton + number_of_wagons * self.weight_empty_wagon_in_ton

            specific_energy_consumption_in_wh_per_gross_ton_km = 1200 * pow(gross_weight_in_ton, -0.62)

            specific_energy_consumption_in_wh_per_gross_ton_km = specific_energy_consumption_in_wh_per_gross_ton_km * trip.slope_profile

            # see page 27ff in the original documentation for additional information on the load factor, empty trip factor and capacity utilization
            load_factor = self.load_per_wagon_in_ton / self.max_load_per_wagon_in_ton
            capacity_utilization = load_factor / (1 + trip.empty_trip_factor)
            if capacity_utilization < 0.6 and trip.during_iteration == 1:
                capacity_utilization = 0.6
            relation_nt_gt = capacity_utilization / (
                    capacity_utilization + (self.weight_empty_wagon_in_ton / self.max_load_per_wagon_in_ton))

            specific_energy_consumption_in_wh_per_net_ton_km = specific_energy_consumption_in_wh_per_gross_ton_km / relation_nt_gt

            energy_in_wh = specific_energy_consumption_in_wh_per_net_ton_km * trip.distance_in_km * net_weight_in_ton

            # apply engine efficiency
            if self.engine_type == "diesel":
                energy_in_wh = energy_in_wh / self.engine_diesel_electric_efficiency

            # convert energy
            energy_in_J = (energy_in_wh / 1000) * 3.6 * pow(10, 6)
            energy_in_MJ = energy_in_J * pow(10, -6)

            # convert energy to emissions
            wtw_emissions_in_kgCO2e = energy_in_MJ * self.ttw_energy_in_MJ_to_wtw_in_gCO2e / 1000
            wtw_emissions_in_tCO2e += wtw_emissions_in_kgCO2e / 1000

            # reduce 'rolling' load
            remaining_load_in_ton -= self.max_load_train_in_ton

        return wtw_emissions_in_tCO2e


class Truck:

    def __init__(self, setting):

        if setting == '40t, diesel, Euro VI' or setting == 'default':
            self.vehicle_type = '40t'
            self.engine_type = 'diesel'
            self.emission_standard = 'Euro VI'
            self.empty_weigth_in_ton = 14
            self.max_weight_in_ton = 40

            # at capacity utilization (in pct.) (Table 22, p. 60)
            self.energy_consumption_in_MJ_per_km = {
                0: 8.2,
                50: 10.4,
                100: 13.3
            }

            # energy-emission factor (DIN EN Norm 16258 - A.4 (7% bio-diesel in volume))
            self.ttw_energy_in_MJ_to_wtw_in_gCO2e = 88.21

        else:
            print('truck type not defined')

        self.max_load_in_ton = self.max_weight_in_ton - self.empty_weigth_in_ton

    def compute_wtw_emissions_in_tCO2e(self, trip):

        wtw_emissions_in_tCO2e = 0

        # this loop handles cases in which the trip encompasses multiple trucks
        remaining_load_in_ton = trip.load_in_ton
        while remaining_load_in_ton > 0:
            load_in_ton = min(remaining_load_in_ton, self.max_load_in_ton)

            load_factor = load_in_ton / self.max_load_in_ton
            capacity_utilization = load_factor / (1 + trip.empty_trip_factor)

            energy_in_MJ_per_km = \
                self.energy_consumption_in_MJ_per_km[0] \
                + capacity_utilization * (
                        self.energy_consumption_in_MJ_per_km[100]
                        - self.energy_consumption_in_MJ_per_km[0]
                )

            energy_in_MJ = energy_in_MJ_per_km * trip.distance_in_km

            # convert energy to emissions
            wtw_emissions_in_kgCO2e = energy_in_MJ * self.ttw_energy_in_MJ_to_wtw_in_gCO2e / 1000
            wtw_emissions_in_tCO2e += wtw_emissions_in_kgCO2e / 1000

            remaining_load_in_ton -= self.max_load_in_ton

        return wtw_emissions_in_tCO2e


class Barge:

    def __init__(self, setting):
        # Tabelle 27 in TREMOD: https://www.ifeu.de/fileadmin/uploads/IFEU-INFRAS-2013-Aktualisierung-der-Emissionsberechnung-f%C3%BCr-die-Binnenschifffahrt-und-%C3%9Cbertragung-der-Daten-in-TREMOD3.pdf
        avg_load_empty = (500 + 500 + 1000 + 1000 + 500) / 5    # = 700
        avg_load_full = (500 + 500 + 1000 + 1000 + 1500) / 5    # = 900

        if setting == 'Europa ship' or setting == 'default':
            self.max_load_in_ton = 1350
            self.installed_power_in_kW = 737
            self.avg_speed_in_kmh = 10.5
            self.specific_diesel_consumption_in_g_kWh = 200
            self.avg_load_in_ton = {'empty': avg_load_empty, 'full': avg_load_full}

        elif setting == 'Grossmotorschiff':
            self.max_load_in_ton = 2500
            self.installed_power_in_kW = 1178
            self.avg_speed_in_kmh = 10.5
            self.specific_diesel_consumption_in_g_kWh = 200
            self.avg_load_in_ton = {'empty': avg_load_empty, 'full': avg_load_full}

        elif setting == 'Jowi class':
            self.max_load_in_ton = 5300
            self.installed_power_in_kW = 2097
            self.avg_speed_in_kmh = 10.5
            self.specific_diesel_consumption_in_g_kWh = 200
            self.avg_load_in_ton = {'empty': avg_load_empty, 'full': avg_load_full}

        else:
            print('barge type not defined')

    def compute_wtw_emissions_in_tCO2e(self, trip):

        wtw_emissions_in_tCO2e = 0

        # this loop handles cases in which the trip encompasses multiple barges
        remaining_load_in_ton = trip.load_in_ton
        while remaining_load_in_ton > 0:
            load_in_ton = min(remaining_load_in_ton, self.max_load_in_ton)

            cargo_utilization = (load_in_ton / self.max_load_in_ton) / (1 + trip.empty_trip_factor)
            if cargo_utilization < 0.6 and trip.during_iteration == 1:
                cargo_utilization = 0.6
            load_factor_empty = self.avg_load_in_ton['empty'] / self.max_load_in_ton
            load_factor_full = self.avg_load_in_ton['full'] / self.max_load_in_ton
            load_factor_cu = load_factor_empty + (load_factor_full - load_factor_empty) * cargo_utilization

            # as in paper page 92
            fuel_consumption_main_engine_in_g_per_ton_km = \
                ((self.installed_power_in_kW * load_factor_cu) / self.avg_speed_in_kmh) \
                / (cargo_utilization * self.max_load_in_ton) \
                * self.specific_diesel_consumption_in_g_kWh

            fuel_consumption_aux_engine_in_g_per_ton_km = 0.05 * fuel_consumption_main_engine_in_g_per_ton_km

            fuel_consumption_total_in_g_per_ton_km = \
                fuel_consumption_main_engine_in_g_per_ton_km \
                + fuel_consumption_aux_engine_in_g_per_ton_km

            # fuel_consumption_total_in_g_per_ton_km = 15
            fuel_consumption_in_kg = fuel_consumption_total_in_g_per_ton_km * load_in_ton * trip.distance_in_km / 1000

            fuel_consumption_in_l = fuel_consumption_in_kg * (1 / 0.9)
            fuel_consumption_in_l_per_ton_km = fuel_consumption_in_l / (250 * 1350 * 0.6)

            # 3.92: WTW CO2e emission factor for marine diesel oil (Table 58, p. 119)
            wtw_emissions_in_tCO2e += fuel_consumption_in_kg * 3.92 / 1000

            # reduce 'rolling' load
            remaining_load_in_ton -= self.max_load_in_ton

        return wtw_emissions_in_tCO2e