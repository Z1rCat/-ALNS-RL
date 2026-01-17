import numpy as np
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    class _Dummy:
        def __getattr__(self, item):
            def _noop(*args, **kwargs):
                return None
            return _noop
    fuzz = _Dummy()
    ctrl = _Dummy()
import matplotlib.pyplot as plt

def fuzzy_HP_all(Cost_preference, Time_preference, Delay_preference, Transshipment_preference, Emissions_preference, actual_Cost, actual_Time, actual_Delay, actual_Transshipment, actual_Emissions):
    # New Antecedent/Consequent objects hold universe variables and membership
    # functions

    Cost = generate_membership_functions('Cost', Cost_universe, Cost_level_values, get_term('Cost')[0], get_term('Cost')[1])

    Emissions = generate_membership_functions('Emissions', Emissions_universe, Emissions_level_values, get_term('Emissions')[0], get_term('Emissions')[1])
    Transshipment = generate_membership_functions('Transshipment', Transshipment_universe, Transshipment_level_values, get_term('Transshipment')[0], get_term('Transshipment')[1])

    Time = generate_membership_functions('Time', Time_universe, Time_level_values, get_term('Time')[0], get_term('Time')[1])

    Delay = generate_membership_functions('Delay', Delay_universe, Delay_level_values, get_term('Delay')[0], get_term('Delay')[1])

    Satisfaction = generate_membership_functions('Satisfaction', Satisfaction_universe, Satisfaction_level_values)

    rule1 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', 1) + '&' + generate_rules_3(Emissions_preference, 'Emissions', 1) + '&' + generate_rules_3(Time_preference, 'Time', 1) + '&' + generate_rules_3(Transshipment_preference, 'Transshipment', 1) + '&' + generate_rules_3(Delay_preference, 'Delay', 1)), Satisfaction['high'])
    rule2 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', 0) + '&' + generate_rules_3(Emissions_preference, 'Emissions', 0) + '&' + generate_rules_3(Time_preference, 'Time', 0) + '&' + generate_rules_3(Transshipment_preference, 'Transshipment', 0) + '&' + generate_rules_3(Delay_preference, 'Delay', 0)), Satisfaction['medium'])
    rule3 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', -1) + '&' + generate_rules_3(Emissions_preference, 'Emissions', -1) + '&' + generate_rules_3(Time_preference, 'Time', -1) + '&' + generate_rules_3(Transshipment_preference, 'Transshipment', -1) + '&' + generate_rules_3(Delay_preference, 'Delay', -1)), Satisfaction['low'])

    # rule1.view()
    satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    satisfactoryping = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)
    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
    satisfactoryping.inputs({'Cost': actual_Cost, 'Emissions': actual_Emissions, 'Time': actual_Time, 'Delay': actual_Delay, 'Transshipment': actual_Transshipment})

    # Crunch the numbers
    satisfactoryping.compute()

    # Satisfaction.view(sim=satisfactoryping)
    return satisfactoryping.output['Satisfaction']

def get_term(attribute):

    if attribute == 'Cost':
        term1, term2 = "'cheap'", "'expensive'"
    elif attribute == 'Emissions':
        term1, term2 = "'low emissions'", "'high emissions'"
    elif attribute == 'Time':
        term1, term2 = "'fast'", "'slow'"
    elif attribute == 'Transshipment':
        term1, term2 = "'low risk'", "'high risk'"
    elif attribute == 'Delay':
        term1, term2= "'reliable'", "'unreliable'"
    return term1, term2

def generate_membership_functions(attribute, universe, level_values,term1=-1, term2=-1):
    if attribute != 'Satisfaction':
        if attribute == 'Emissions' and three_eco_labels == 1:
            Attribute = ctrl.Antecedent(universe, attribute)
            min_value, define_L1_value, define_L1_value2, define_L3_value, define_L3_value2, define_L3_value3, define_L3_value4, define_L5_value, define_L5_value2, max_value = level_values

            Attribute[eval(term1)] = fuzz.trapmf(Attribute.universe,[min_value, min_value, define_L1_value, define_L1_value2])
            Attribute['medium'] = fuzz.trapmf(Attribute.universe, [define_L3_value, define_L3_value2, define_L3_value3, define_L3_value4])
            Attribute[eval(term2)] = fuzz.trapmf(Attribute.universe,[define_L5_value, define_L5_value2, max_value, max_value])


        else:
            # New Antecedent/Consequent objects hold universe variables and membership
            # functions

            Attribute = ctrl.Antecedent(universe, attribute)

            # Auto-membership function population is possible with .automf(3, 5, or 7)
            min_value, define_L1_value, define_L1_value2, define_L2_value, define_L2_value2, define_L2_value3, define_L3_value, define_L3_value2, define_L3_value3, define_L4_value, define_L4_value2, define_L4_value3, define_L5_value, define_L5_value2, max_value = level_values
            Attribute['very ' + eval(term1)] = fuzz.trapmf(Attribute.universe,[min_value, min_value, define_L1_value, define_L1_value2])
            Attribute[eval(term1)] = fuzz.trimf(Attribute.universe, [define_L2_value, define_L2_value2, define_L2_value3])
            Attribute['medium'] = fuzz.trimf(Attribute.universe, [define_L3_value, define_L3_value2, define_L3_value3])
            Attribute[eval(term2)] = fuzz.trimf(Attribute.universe, [define_L4_value, define_L4_value2, define_L4_value3])
            Attribute['very ' + eval(term2)] = fuzz.trapmf(Attribute.universe,[define_L5_value, define_L5_value2, max_value, max_value])

            #Transshipment is not fuzzy
            # else:
            #     #Transshipment
            #     min_value, define_L1_value, define_L2_value, define_L3_value, max_value = level_values
            #     Attribute['very ' + eval(term1)] = fuzz.trapmf(Attribute.universe,
            #                                                    [min_value, define_L1_value, define_L2_value])
            #     Attribute[eval(term1)] = fuzz.trimf(Attribute.universe, [define_L1_value, define_L2_value, define_L3_value])
            #     Attribute['very ' + eval(term2)] = fuzz.trapmf(Attribute.universe,
            #                                                    [define_L2_value, define_L3_value, max_value, max_value])
        return Attribute
    else:
        #Satisfaction
        low_value1, low_value2, medium_value1, medium_value2, medium_value3, high_value1, high_value2 = level_values
        Satisfaction = ctrl.Consequent(universe, attribute, defuzzify_method='centroid')
        Satisfaction['low'] = fuzz.trimf(Satisfaction.universe, [low_value1, low_value1, low_value2])
        Satisfaction['medium'] = fuzz.trimf(Satisfaction.universe, [medium_value1, medium_value2, medium_value3])
        Satisfaction['high'] = fuzz.trimf(Satisfaction.universe, [high_value1, high_value2, high_value2])

        # You can see how these look with .view()
        # Attribute.view()
        # emission.view()
        # Satisfaction.view()
        return Satisfaction

def generate_rules_3_copy(preferred_level, attribute, satisfy):
    term1, term2 = get_term(attribute)
    if preferred_level == 1:
        if satisfy == 1 or satisfy == 0:
            return '(' + attribute + "['very ' + term1]" + ')'
        else:
            return '(' + attribute + "[term1]" + ' | ' + attribute + "[term2]" + ' | ' + attribute + "['very ' + term2]" + ')'
    elif preferred_level == 2:
        if satisfy == 1:
            return '(' + attribute + "['very ' + term1]" + ')'
        elif satisfy == 0:
            return '(' + attribute + "[term1]" + ')'
        else:
            return '(' + attribute + "[term2]" + ' | ' + attribute + "['very ' + term2]" + ')'
    elif preferred_level == 3:
        if satisfy == 1:
            return '(' + attribute + "['very ' + term1]" + ' | ' + attribute + "[term1]" + ')'
        elif satisfy == 0:
            return '(' + attribute + "[term2]" + ')'
        else:
            return '(' + attribute + "['very ' + term2]" + ')'
    else:
        if satisfy == 1 or satisfy == 0:
            return '(' + attribute + "['very ' + term1]" + ' | ' + attribute + "[term1]" + ' | ' + attribute + "[term2]" + ')'
        else:
            return '(' + attribute + "['very ' + term2]"  + ')'


def generate_rules_3(preferred_level, attribute, satisfy):
    term1, term2 = get_term(attribute)
    if attribute == 'Emissions' and three_eco_labels == 1:
        if preferred_level == 1:
            if satisfy == 1:
                return '(' + attribute + "[" + term1 + "]" + ')'
            else:
                # satisfy can only == -1 because it's the first level
                return '(' + attribute + "[" + "'medium'" + "]" + ' | ' + attribute + "[" + term2 + "]" + ')'
        elif preferred_level == 2:
            if satisfy == 1:
                return '(' + attribute + "[" + term1 + "]" + ')'
            elif satisfy == 0:
                return '(' + attribute + "[" + "'medium'" + "]"  + ')'
            else:
                return '(' + attribute + "[" + term2 + "]" + ')'
        else:
            if satisfy == 1:
                return '(' + attribute + "[" + term1 + "]" + ' | ' + attribute + "[" + "'medium'" + "]" + ')'
            else:
                #satisfy can only == 0 because it's the last level
                return '(' + attribute + "[" + term2 + "]"  + ')'
    else:
        if preferred_level == 1:
            if satisfy == 1:
                return '(' + attribute + "['very '" + term1 + "]" + ')'
            else:
                # satisfy can only == -1 because it's the first level
                return '(' + attribute + "[" + term1 + "]" + ' | ' + attribute + "[" + "'medium'" + "]" + ' | ' + attribute + "[" + term2 + "]" + ' | ' + attribute + "['very ' + " + term2 + "]" + ')'
        elif preferred_level == 2:
            if satisfy == 1:
                return '(' + attribute + "['very ' + " + term1 + "]" + ')'
            elif satisfy == 0:
                return '(' + attribute + "[" + term1 + "]" + ')'
            else:
                return '(' + attribute + "[" + "'medium'" + "]" + ' | ' + attribute + "[" + term2 + "]" + ' | ' + attribute + "['very ' + " + term2 + "]" + ')'
        elif preferred_level == 3:
            if satisfy == 1:
                return '(' + attribute + "['very ' + " + term1 + "]" + ' | ' + attribute + "[" + term1 + "]" + ')'
            elif satisfy == 0:
                return '(' + attribute + "[" + "'medium'" + "]"  + ')'
            else:
                return '(' + attribute + "[" + term2 + "]" + ' | ' + attribute + "['very ' + " + term2 + "]" + ')'
        elif preferred_level == 4:
            if satisfy == 1:
                return '(' + attribute + "['very ' + " + term1 + "]" + ' | ' + attribute + "[" + term1 + "]" + ' | ' + attribute + "[" + "'medium'" + "]" + ')'
            elif satisfy == 0:
                return '(' + attribute + "[" + term2 + "]" + ')'
            else:
                #satisfy can only == 0 because it's the last level
                return '(' + attribute + "['very ' + " + term2 + "]"  + ')'
        else:
            if satisfy == 1:
                return '(' + attribute + "['very ' + " + term1 + "]" + ' | ' + attribute + "[" + term1 + "]" + ' | ' + attribute + "[" + "'medium'" + "]" + ' | '+ attribute + "[" + term2 + "]" + ')'
            else:
                #satisfy can only == 0 because it's the last level
                return '(' + attribute + "['very ' + " + term2 + "]"  + ')'

def get_rules(preference, attribute, real_attribute, Satisfaction):
    locals()[attribute] = real_attribute
    rule1 = ctrl.Rule(eval(generate_rules_3(preference, attribute, 1)), Satisfaction['high'])
    if preference != 1:
        # when it's the first level, no medium Satisfaction
        rule2 = ctrl.Rule(eval(generate_rules_3(preference, attribute, 0)), Satisfaction['medium'])
    else:
        rule2 = -1

    if attribute == 'Emissions' and three_eco_labels == 1:
        if preference != 3:
            # when it's the last level, no low Satisfaction
            rule3 = ctrl.Rule(eval(generate_rules_3(preference, attribute, -1)), Satisfaction['low'])
        else:
            rule3 = -1
    else:
        if preference != 5:
            # when it's the last level, no low Satisfaction
            rule3 = ctrl.Rule(eval(generate_rules_3(preference, attribute, -1)), Satisfaction['low'])
        else:
            rule3 = -1
    rules = [rule1, rule2, rule3]
    rules = [x for x in rules if not isinstance(x, int)]
    return rules

def fuzzy_HP_one(attribute,  actual_value, preference):
    if attribute == 'Cost':
        universe, level_values = Cost_universe, Cost_level_values
    elif attribute == 'Emissions':
        universe, level_values = Emissions_universe, Emissions_level_values
    elif attribute == 'Time':
        universe, level_values = Time_universe, Time_level_values
    elif attribute == 'Delay':
        universe, level_values = Delay_universe, Delay_level_values
    else:
        universe, level_values = Transshipment_universe, Transshipment_level_values
    term1, term2 = get_term(attribute)
    locals()[attribute] = generate_membership_functions(attribute, universe, level_values, term1, term2)
    # locals()[attribute].view()
    # plt.savefig(
    #     "C:/Users/yimengzhang/OneDrive/桌面/My papers/IAME hetegenous preferences/Figures/satisfactory_changes_new_newMF/" + "membership_function" + attribute + '.pdf',
    #     format='pdf')
    # plt.close()
    Satisfaction = generate_membership_functions('Satisfaction', Satisfaction_universe, Satisfaction_level_values)
    # Satisfaction.view()
    # plt.savefig(
    #     "C:/Users/yimengzhang/OneDrive/桌面/My papers/IAME hetegenous preferences/Figures/satisfactory_changes_new_newMF/" + "membership_function" + 'satisfaction' + '.pdf',
    #     format='pdf')

    # plt.close()
    #the rules are different for different requests
    #general rule: all preferred levels are met, then high; high preferred levels are met, then medium; only low preferred levels are met, then low
    def generate_rules(preferred_level, attribute, satisfy):
        if preferred_level == 1:
            if satisfy == 1:
                return attribute + "['very ' + term1]"
            else:
                return attribute + "[term1]" + ' | ' + attribute + "[term2]" + ' | ' + attribute + "['very ' + term2]"
        elif preferred_level == 2:
            return attribute + "['very ' + term1]" + ' | ' + attribute + "[term1]"


    # rule1 = ctrl.Rule(eval(generate_rules(preference, 'Cost', 1)), Satisfaction['satisfied'])
    # rule2 = ctrl.Rule(eval(generate_rules(preference, 'Cost', 0)), Satisfaction['unsatisfied'])

    #1 means higher level, 0 same level, -1 lower level
    rules = get_rules(preference, attribute, locals()[attribute], Satisfaction)


    # rule1.view()
    # satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2])
    satisfactoryping_ctrl = ctrl.ControlSystem(rules)
    # if preference != 1 and preference != 5:
    #     satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    # elif preference == 1:
    #     satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule3])
    # else:
    #     satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2])

    satisfactoryping = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)
    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
    satisfactoryping.input[attribute] = actual_value
    # satisfactoryping.input['emission'] = 9.8

    # Crunch the numbers
    # try:
    satisfactoryping.compute()
    # except:
    #     print(0)
    # Satisfaction.view(sim=satisfactoryping)

    return satisfactoryping.output['Satisfaction']

def get_preference_string(attribute,preferred_level):
    term1, term2 = get_term(attribute)
    term1, term2 = eval(term1), eval(term2)
    if preferred_level == 1:
        preference = 'very ' + term1

    elif preferred_level == 2:
        preference = term1
    elif preferred_level == 3:
        preference = 'medium'
    elif preferred_level == 4:
        preference = term2
    else:
        preference = 'very ' + term2

    return preference, term1, term2
def plot_changes(attribute, universe, level_values):

    for preferred_level in [1,2,3,4,5]:
        satisfactory_results = []
        for actual_value in universe:
            satisfactory_results.append(fuzzy_HP_one(attribute, actual_value,preferred_level))
        plt.scatter(universe, satisfactory_results)
        satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3, satisfactory_value_hard_L4, satisfactory_value_hard_L5, satisfactory_value_hard_last = 0, 0, 0, 0, 0, 0

        min_value, define_L1_value, define_L2_value, define_L3_value, define_L4_value, define_L5_value, max_value = level_values[0], level_values[1], level_values[4], level_values[7], level_values[10], level_values[13], level_values[14]
        preference, term1, term2 = get_preference_string(attribute, preferred_level)
        if preferred_level == 1:
            satisfactory_value_hard_L1 = 100
        elif preferred_level == 2:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2 = 100, 50
        elif preferred_level == 3:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3 = 100, 100, 50
        elif preferred_level == 4:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3, satisfactory_value_hard_L4 = 100, 100, 100, 50
        else:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3, satisfactory_value_hard_L4, satisfactory_value_hard_L5 = 100, 100, 100, 100, 50
        plt.plot([min_value,define_L1_value],[satisfactory_value_hard_L1,satisfactory_value_hard_L1], 'c', label= 'very ' + term1)
        plt.plot([define_L1_value,define_L2_value],[satisfactory_value_hard_L2,satisfactory_value_hard_L2], 'g', label= term1)
        plt.plot([define_L2_value, define_L3_value], [satisfactory_value_hard_L3, satisfactory_value_hard_L3], 'b', label='medium')
        plt.plot([define_L3_value,define_L4_value],[satisfactory_value_hard_L4,satisfactory_value_hard_L4], 'm', label= term2)
        plt.plot([define_L4_value, define_L5_value], [satisfactory_value_hard_L5, satisfactory_value_hard_L5], 'y', label='very ' + term2)
        plt.plot([define_L5_value,max_value],[satisfactory_value_hard_last,satisfactory_value_hard_last], 'r', label= 'very ' + term2)

        plt.legend()
        if attribute == 'Cost':
            plt.xlabel('Cost (euro/km/TEU)')
        elif attribute == 'Emissions':
            plt.xlabel('Emissions (kg/km/TEU)')
        elif attribute == 'Time':
            plt.xlabel('Time (h)')
        elif attribute == 'Transshipment':
            plt.xlabel('Number of Transshipments')
        elif attribute == 'Delay':
            plt.xlabel('Delay (h)')
        plt.ylabel('Satisfaction value')
        plt.title('Hard (lines) vs. fuzzy (dots) constraints (preference: ' + preference + ')')
        # plt.show()
        plt.savefig("C:/Users/yimengzhang/OneDrive/桌面/My papers/IAME hetegenous preferences/Figures/satisfactory_changes/" + attribute + 'preference_test' + str(preferred_level) + '.pdf',
                    format='pdf')
        plt.close()

def plot_changes_all():
    plot_changes('Cost', Cost_universe, Cost_level_values)
    plot_changes('Time', Time_universe, Time_level_values)
    plot_changes('Emissions', Emissions_universe, Emissions_level_values)
    plot_changes('Transshipment', Transshipment_universe, Transshipment_level_values)
    plot_changes('Delay', Delay_universe, Delay_level_values)

def plot_changes_overall():
    # for universe, level_values in [Cost_universe, Time_universe]
    Time_preference, Delay_preference, Transshipment_preference, Emissions_preference, actual_Time, actual_Delay, actual_Transshipment, actual_Emissions = 4,4,4,4,0.1,0.1,0,0.1
    attribute = 'Cost'
    universe = np.arange(0, 1.6, 0.1)
    for preferred_level in [1,2,3,4]:
        satisfactory_results = []
        for actual_value in universe:
            satisfactory_results.append(fuzzy_HP_all(preferred_level, Time_preference, Delay_preference, Transshipment_preference, Emissions_preference, actual_value, actual_Time, actual_Delay, actual_Transshipment, actual_Emissions))
        plt.scatter(universe, satisfactory_results)
        satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3, satisfactory_value_hard_L4 = 0, 0, 0, 0
        preference, term1, term2 = get_preference_string(attribute, preferred_level)
        if preferred_level == 1:
            satisfactory_value_hard_L1 = 100
        elif preferred_level == 2:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2 = 100, 100
        elif preferred_level == 3:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3 = 100, 100, 100
        else:
            satisfactory_value_hard_L1, satisfactory_value_hard_L2, satisfactory_value_hard_L3, satisfactory_value_hard_L4 = 100, 100, 100, 100
        min_value, define_L1_value, define_L2_value, define_L3_value, define_L4_value, max_value = level_values

        plt.plot([min_value,define_L1_value],[satisfactory_value_hard_L1,satisfactory_value_hard_L1], 'g', label= 'very ' + term1)
        plt.plot([define_L1_value,define_L2_value],[satisfactory_value_hard_L2,satisfactory_value_hard_L2], 'b', label= term1)
        plt.plot([define_L2_value,define_L3_value],[satisfactory_value_hard_L3,satisfactory_value_hard_L3], 'y', label= term2)
        plt.plot([define_L3_value,max_value],[satisfactory_value_hard_L4,satisfactory_value_hard_L4], 'r', label= 'very ' + term2)
        plt.legend()
        if attribute == 'Cost':
            plt.xlabel('Cost (euro/km/TEU)')
        elif attribute == 'Emissions':
            plt.xlabel('Emissions (kg/km/TEU)')
        elif attribute == 'Time':
            plt.xlabel('Time (h)')
        elif attribute == 'Transshipment':
            plt.xlabel('Number of Transshipments')
        elif attribute == 'Delay':
            plt.xlabel('Delay (h)')
        plt.ylabel('Satisfaction value')
        plt.title('Hard (lines) vs. fuzzy (dots) constraints (preference: ' + preference + ')')
        # plt.show()
        plt.savefig("C:/Users/yimengzhang/OneDrive/桌面/My papers/IAME hetegenous preferences/Figures/satisfactory_changes_overall/" + attribute + 'preference' + str(preferred_level) + '.pdf',
                    format='pdf')
        plt.close()

def two_input_3D(Cost_preference, Time_preference, Delay_preference, Transshipment_preference, Emissions_preference, actual_Cost, actual_Time, actual_Delay, actual_Transshipment, actual_Emissions):
    # Sparse universe makes calculations faster, without sacrifice accuracy.
    # Only the critical points are included here; making it higher resolution is
    # unnecessary.

    Cost = generate_membership_functions('Cost', Cost_universe, Cost_level_values, get_term('Cost')[0], get_term('Cost')[1])

    Emissions = generate_membership_functions('Emissions', Emissions_universe, Emissions_level_values, get_term('Emissions')[0], get_term('Emissions')[1])
    # Transshipment = generate_membership_functions('Transshipment', Transshipment_universe, Transshipment_level_values, get_term('Transshipment')[0], get_term('Transshipment')[1])
    #
    # Time = generate_membership_functions('Time', Time_universe, Time_level_values, get_term('Time')[0], get_term('Time')[1])
    #
    # Delay = generate_membership_functions('Delay', Delay_universe, Delay_level_values, get_term('Delay')[0], get_term('Delay')[1])

    Satisfaction = generate_membership_functions('Satisfaction', Satisfaction_universe, Satisfaction_level_values)

    rules_cost = get_rules(Cost_preference, 'Cost', Cost, Satisfaction)
    rules_emissions = get_rules(Emissions_preference, 'Emissions', Emissions, Satisfaction)
    rules = rules_cost + rules_emissions
    # rule1.view()
    satisfactoryping_ctrl = ctrl.ControlSystem(rules)
    # Later we intend to run this system with a 17*17 set of inputs, so we allow
    # that many plus one unique runs before results are flushed.
    # Subsequent runs would return in 1/8 the time!
    max_value_cost = Cost_level_values[-1]
    max_value_emissions = Emissions_level_values[-1]
    scatters_number_cost = int(max_value_cost*10 + 1)
    scatters_number_emissions = int(max_value_emissions * 10 + 1)
    sim = ctrl.ControlSystemSimulation(satisfactoryping_ctrl, flush_after_run=scatters_number_cost * scatters_number_emissions + 1)
    # We can simulate at higher resolution with full accuracy
    upsampled_cost = np.linspace(0, max_value_cost, scatters_number_cost)
    upsampled_emissions = np.linspace(0, max_value_emissions, scatters_number_emissions)
    x, y = np.meshgrid(upsampled_cost, upsampled_emissions)
    # z = np.zeros_like
    z = np.array(np.empty(shape=(scatters_number_emissions,scatters_number_cost)))
    # Loop through the
    # system 17*17 times to collect the control surface
    for i in range(scatters_number_cost):
        for j in range(scatters_number_emissions):
            sim.input['Cost'] = x[j, i]
            sim.input['Emissions'] = y[j, i]
            # try:
            sim.compute()
            # except:
            #     print(1)
            z[j, i] = sim.output['Satisfaction']

    # Plot the result in pretty 3D with alpha blending

    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True, shade = False)

    #shades
    # cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    # cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
    # cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

    #first parameter control up/down, second left/right
    ax.view_init(25, 45)
    ax.set_xlabel('Cost (euro/km/TEU)')
    ax.set_ylabel('Emissions (kg/km/TEU)')
    ax.set_zlabel('Satisfaction')
    cost_preference = get_preference_string('Cost', Cost_preference)[0]
    emissions_preference = get_preference_string('Emissions', Emissions_preference)[0]
    plt.title("Satisfaction changes (Cost: " + cost_preference + ', Emissions: ' + emissions_preference + ')')
    plt.savefig(
        "C:/Users/yimengzhang/OneDrive/桌面/My papers/Collaborative planning for sustainable intermodal transport/Figures/" + "3D Satisfaction changes (Cost" + cost_preference + 'Emissions' + emissions_preference + ')' + '.pdf',
        format='pdf')
    plt.close()
    # plt.show()
    # print(2)


def five_attributes_to_satisfactory(Cost_preference, Time_preference, Delay_preference, Transshipment_preference, Emissions_preference, actual_Cost, actual_Time, actual_Delay, actual_Transshipment, actual_Emissions,only_emissions=0):
    # Sparse universe makes calculations faster, without sacrifice accuracy.
    # Only the critical points are included here; making it higher resolution is
    # unnecessary.

    Cost = generate_membership_functions('Cost', Cost_universe, Cost_level_values, get_term('Cost')[0], get_term('Cost')[1])

    Emissions = generate_membership_functions('Emissions', Emissions_universe, Emissions_level_values, get_term('Emissions')[0], get_term('Emissions')[1])


    Satisfaction = generate_membership_functions('Satisfaction', Satisfaction_universe, Satisfaction_level_values)


    rules_cost = get_rules(Cost_preference, 'Cost', Cost, Satisfaction)
    rules_emissions = get_rules(Emissions_preference, 'Emissions', Emissions, Satisfaction)
    #
    # rule1 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', 1)), Satisfaction['high'])
    # rule2 = ctrl.Rule(eval(generate_rules_3(Emissions_preference, 'Emissions', 1)), Satisfaction['high'])
    # rule3 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', 0)), Satisfaction['medium'])
    # rule4 = ctrl.Rule(eval(generate_rules_3(Emissions_preference, 'Emissions', 0)), Satisfaction['medium'])
    # rule5 = ctrl.Rule(eval(generate_rules_3(Cost_preference, 'Cost', -1)), Satisfaction['low'])
    # rule6 = ctrl.Rule(eval(generate_rules_3(Emissions_preference, 'Emissions', -1)), Satisfaction['low'])
    if only_emissions == 1:
        # rules = rules_cost + rules_emissions
        rules = rules_emissions
        satisfactoryping_ctrl = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)
        # sim.inputs({'Cost': actual_Cost, 'Emissions': actual_Emissions})
        sim.inputs({'Emissions': actual_Emissions})
        sim.compute()
        # print(sim.output['Satisfaction'])
        return sim.output['Satisfaction']
    else:

        Transshipment = generate_membership_functions('Transshipment', Transshipment_universe,Transshipment_level_values, get_term('Transshipment')[0],get_term('Transshipment')[1])
        Time = generate_membership_functions('Time', Time_universe, Time_level_values, get_term('Time')[0],get_term('Time')[1])
        Delay = generate_membership_functions('Delay', Delay_universe, Delay_level_values, get_term('Delay')[0],get_term('Delay')[1])
        rules_Transshipment = get_rules(Transshipment_preference, 'Transshipment', Transshipment, Satisfaction)
        rules_Time = get_rules(Time_preference, 'Time', Time, Satisfaction)
        rules_Delay = get_rules(Delay_preference, 'Delay', Delay, Satisfaction)
        # rule7 = ctrl.Rule(eval(generate_rules_3(Time_preference, 'Time', 1)), Satisfaction['high'])
        # rule8 = ctrl.Rule(eval(generate_rules_3(Delay_preference, 'Delay', 1)), Satisfaction['high'])
        # rule9 = ctrl.Rule(eval(generate_rules_3(Time_preference, 'Time', 0)), Satisfaction['medium'])
        # rule10 = ctrl.Rule(eval(generate_rules_3(Delay_preference, 'Delay', 0)), Satisfaction['medium'])
        # rule11 = ctrl.Rule(eval(generate_rules_3(Time_preference, 'Time', -1)), Satisfaction['low'])
        # rule12 = ctrl.Rule(eval(generate_rules_3(Emissions_preference, 'Emissions', -1)), Satisfaction['low'])
        #
        # rule13 = ctrl.Rule(eval(generate_rules_3(Transshipment_preference, 'Transshipment', 1)), Satisfaction['high'])
        # rule14 = ctrl.Rule(eval(generate_rules_3(Transshipment_preference, 'Transshipment', 0)), Satisfaction['medium'])
        # rule15 = ctrl.Rule(eval(generate_rules_3(Transshipment_preference, 'Transshipment', -1)), Satisfaction['low'])
        rules = rules_cost + rules_emissions + rules_Transshipment + rules_Time + rules_Delay
        # satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
        satisfactoryping_ctrl = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)

        sim.inputs({'Cost': actual_Cost, 'Time': actual_Time, 'Delay': actual_Delay, 'Transshipment': actual_Transshipment, 'Emissions': actual_Emissions})
        sim.compute()
        # print(sim.output['Satisfaction'])
        return sim.output['Satisfaction']
def get_value_from_preference(attribute, preference):
    if attribute == 'Cost':
        return Cost_level_values_actual[preference]
    elif attribute == 'Emissions':
        return Emissions_level_values_actual[preference]
    elif attribute == 'Time':
        return Time_level_values_actual[preference]
    elif attribute == 'Delay':
        return Delay_level_values_actual[preference]
    else:
        return Transshipment_level_values_actual[preference]


# Cost_universe = np.arange(0, 1.6, 0.01)
# Cost_level_values = [0, 0.28, 0.43, 0.49, 0.75,1, 1.6]
# Emissions_universe = np.arange(0, 1.6, 0.01)
# Emissions_level_values = [0,0.25,0.33,0.5, 0.91,1.3,1.6]
# min_value, define_L1_value, define_L1_value2, define_L2_value, define_L2_value2, define_L2_value3, define_L3_value, define_L3_value2, define_L3_value3, define_L4_value, define_L4_value2, define_L4_value3, define_L5_value, define_L5_value2, max_value
Cost_universe = np.arange(0, 1.8, 0.01)
Cost_level_values = [0, 0.3, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 1.1, 1.0, 1.2, 1.4, 1.3, 1.5, 1.8]
Cost_level_values_actual = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
#the following will have no intersection between two levels, but the connect point of two triangles will have an error when compute satisfaction values because it has 0 membership to all levels
# Cost_level_values = [0, 0.3, 0.45, 0.45, 0.6, 0.75, 0.75, 0.9, 1.05, 1.05, 1.2, 1.35, 1.35, 1.5, 1.8]
emission_by_Arne = 0
three_eco_labels = 0
if emission_by_Arne == 1:
    #1-5: 10 gCOe2/tkm, 20, 37, 52, >52
    #convert to transport unit: 13 ton per unit
    #130 gCOe2/TUkm, 260, 481, 676, >676
    Emissions_universe = np.arange(0, 1.0, 0.001)
    # (0.481 - 0.26) / 3 + 0.26
    # Out[3]: 0.33366666666666667
    # (0.481 - 0.26) / 3 * 2 + 0.26
    # Out[4]: 0.4073333333333333
    # (0.676 - 0.481) / 3 + 0.481
    # Out[5]: 0.546
    # (0.676 - 0.481) / 3 * 2 + 0.481
    # Out[6]: 0.611
    if three_eco_labels == 1:
        Emissions_level_values = [0, 0.39, 0.52, 0.39, 0.52, 0.65, 0.78, 0.65, 0.78, 1.0]
        Emissions_level_values_actual = [0, 0.39, 0.65, 0.65]
    else:
        Emissions_level_values = [0, 0.13, 0.217, 0.173, 0.26, 0.407, 0.334, 0.481, 0.611, 0.546, 0.676, 0.676, 0.676, 0.676, 1.0]
        Emissions_level_values_actual = [0, 0.13, 0.26, 0.481, 0.676, 0.676]
else:
    Emissions_universe = np.arange(0, 1.8, 0.01)
    Emissions_level_values = [0, 0.3, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 1.1, 1.0, 1.2, 1.4, 1.3, 1.5, 1.8]
    Emissions_level_values_actual = [0, 0.3, 0.6, 0.9, 1.2, 1.5]

Time_universe = np.arange(0, 2.2, 0.01)
Time_level_values = [0, 0.5, 0.7, 0.6, 0.8, 1, 0.9, 1.1, 1.3, 1.2, 1.4, 1.6, 1.5, 1.7, 2.2]
Time_level_values_actual = [0, 0.5, 0.8, 1.1, 1.4, 1.7]

Delay_universe = np.arange(0, 0.15, 0.001)
Delay_level_values = [0,0.01,0.03,0.02,0.04,0.06,0.05,0.07,0.09,0.08,0.1,0.12,0.11,0.13,0.15]
Delay_level_values_actual = [0, 0.01, 0.04, 0.07, 0.1, 0.13]

Transshipment_universe = np.arange(0, 150, 1)
Transshipment_level_values = [0,10,30,20,40,60,50,70,90,80,100,120,110,130,150]
Transshipment_level_values_actual = [0, 10, 40, 70, 100, 130]

sensitivity_analysis = 0
if sensitivity_analysis == 1:
    deviation = 0.01
    def deviate(deviation, universe, level_values, level_values_actual):
        deviation_value = level_values[-1] * deviation
        # universe = universe + deviation_value
        # universe[0] = 0
        universe = np.arange(0, universe[-1] + deviation_value, universe[-1] - universe[-2])
        level_values = list(np.array(level_values) + deviation_value)
        level_values[0] = 0
        level_values_actual = list(np.array(level_values_actual) + deviation_value)
        level_values_actual[0] = 0
        return universe, level_values, level_values_actual
    Cost_universe, Cost_level_values, Cost_level_values_actual = deviate(deviation, Cost_universe, Cost_level_values, Cost_level_values_actual)
    Emissions_universe, Emissions_level_values, Emissions_level_values_actual = deviate(deviation, Emissions_universe, Emissions_level_values, Emissions_level_values_actual)
    Time_universe, Time_level_values, Time_level_values_actual = deviate(deviation, Time_universe, Time_level_values,
                                                                         Time_level_values_actual)
    Delay_universe, Delay_level_values, Delay_level_values_actual = deviate(deviation, Delay_universe, Delay_level_values,
                                                                         Delay_level_values_actual)
    Transshipment_universe, Transshipment_level_values, Transshipment_level_values_actual = deviate(deviation, Transshipment_universe, Transshipment_level_values,
                                                                         Transshipment_level_values_actual)



Satisfaction_universe = np.arange(0, 100, 1)
Satisfaction_level_values = [0, 30, 20, 50, 80, 70, 100]

only_emissions = 0

if __name__ == '__main__':
    #
    # Transshipment_universe = np.arange(0, 5, 1)
    # Transshipment_level_values = [0,1,2,3,4,5]

    # fuzzy_HP_one('Delay', 1, 2)
    # fuzzy_HP_one('Time',  0.3, 2)
    # fuzzy_HP_one('Emissions',  0.3, 2)
    # fuzzy_HP_one('Cost',  0.59, 2)
    # fuzzy_HP_one('Transshipment', 10, 2)
    # print(fuzzy_HP_all(4, 4, 4, 3, 4, 0.1, 0.1, 0.1, 0, 0.1))
    plot_changes_all()
    # plot_changes('Cost', Cost_universe, Cost_level_values)
    # plot_changes_overall()
    two_input_3D(5, 5, 5, 5, 5, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(1, 5, 5, 5, 1, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(2, 5, 5, 5, 2, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(3, 5, 5, 5, 3, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(4, 5, 5, 5, 4, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(1, 5, 5, 5, 5, 0.1, 0.1, 0.1, 0, 0.1)
    two_input_3D(5, 5, 5, 5, 1, 0.1, 0.1, 0.1, 0, 0.1)

    # five_attributes_to_satisfactory(3, 1, 1, 1, 1, 0.6, 0.6, 1, 20, 0.5)
    # five_attributes_to_satisfactory(1, 5, 5, 5, 1, 0.1, 0.1, 0.1, 0, 0.1)
    # five_attributes_to_satisfactory(2, 5, 5, 5, 2, 0.1, 0.1, 0.1, 0, 0.1)
    # five_attributes_to_satisfactory(3, 5, 5, 5, 3, 0.1, 0.1, 0.1, 0, 0.1)
    # five_attributes_to_satisfactory(4, 5, 5, 5, 4, 0.1, 0.1, 0.1, 0, 0.1)
    # five_attributes_to_satisfactory(1, 5, 5, 5, 5, 0.1, 0.1, 0.1, 0, 0.1)
    # five_attributes_to_satisfactory(5, 5, 5, 5, 1, 0.1, 0.1, 0.1, 0, 0.1)
    # print(2)


# def fuzzy_HP(r):
#     cost_preference, speed_preference, delay_preference, transshipment_preference, emissions_preference = R_info[r]
#     # New Antecedent/Consequent objects hold universe variables and membership
#     # functions
#     cost = ctrl.Antecedent(np.arange(0, 1.6, 0.1), 'cost')
#     emission = ctrl.Antecedent(np.arange(0, 1.6, 0.1), 'emission')
#     transshipment = ctrl.Antecedent(np.arange(0, 6, 1), 'transshipment')
#     delay = ctrl.Antecedent(np.arange(0, 6, 1), 'delay')
#     speed = ctrl.Antecedent(np.arange(0, 131, 1), 'speed')
#
#     satisfactory = ctrl.Consequent(Satisfaction_universe, 'satisfactory')
#
#     # Auto-membership function population is possible with .automf(3, 5, or 7)
#     # cost
#     cost['Level1'] = fuzz.trapmf(cost.universe, [0, 0, 0.28, 0.43])
#     cost['Level2'] = fuzz.trimf(cost.universe, [0.28, 0.43, 0.75])
#     cost['Level3'] = fuzz.trimf(cost.universe, [0.43, 0.75, 1])
#     cost['Level4'] = fuzz.trapmf(cost.universe, [0.75, 1, 1.5, 1.5])
#     # emission
#     emission['Level1'] = fuzz.trapmf(emission.universe, [0, 0, 0.25, 0.33])
#     emission['Level2'] = fuzz.trimf(emission.universe, [0.25, 0.33, 0.91])
#     emission['Level3'] = fuzz.trimf(emission.universe, [0.33, 0.91, 1.3])
#     emission['Level4'] = fuzz.trapmf(emission.universe, [0.91, 1.3, 1.5, 1.5])
#     # transshipment
#     transshipment['Level1'] = fuzz.trapmf(transshipment.universe, [0, 0, 1, 2])
#     transshipment['Level2'] = fuzz.trimf(transshipment.universe, [1, 2, 3])
#     transshipment['Level3'] = fuzz.trimf(transshipment.universe, [2, 3, 4])
#     transshipment['Level4'] = fuzz.trapmf(transshipment.universe, [3, 4, 5, 5])
#     # delay
#     delay['Level1'] = fuzz.trapmf(delay.universe, [0, 0, 1, 2])
#     delay['Level2'] = fuzz.trimf(delay.universe, [1, 2, 3])
#     delay['Level3'] = fuzz.trimf(delay.universe, [2, 3, 4])
#     delay['Level4'] = fuzz.trapmf(delay.universe, [3, 4, 5, 5])
#     # speed
#     speed['Level4'] = fuzz.trapmf(speed.universe, [0, 0, 10, 40])
#     speed['Level3'] = fuzz.trimf(speed.universe, [10, 40, 70])
#     speed['Level2'] = fuzz.trimf(speed.universe, [40, 70, 100])
#     speed['Level1'] = fuzz.trapmf(speed.universe, [70, 100, 130, 130])
#
#     #satisfactory
#     satisfactory['low'] = fuzz.trimf(satisfactory.universe, [0, 0, 50])
#     satisfactory['medium'] = fuzz.trimf(satisfactory.universe, [0, 50, 100])
#     satisfactory['high'] = fuzz.trimf(satisfactory.universe, [50, 50, 100])
#
#     # You can see how these look with .view()
#     cost['average'].view()
#     emission.view()
#     satisfactory.view()
#
#     #the rules are different for different requests
#     #general rule: all preferred levels are met, then high; high preferred levels are met, then medium; only low preferred levels are met, then low
#     def generate_rules(preferred_level, attribute):
#         if preferred_level == 1:
#             return attribute + "['Level1']"
#         elif preferred_level == 2:
#             return attribute + "['Level1']" + ' | ' + attribute + "['Level2']"
#
#     if cost_preference == 1 and emissions_preference == 1:
#         # if
#         rule1 = ctrl.Rule(cost['Level3'] | emission['poor'], satisfactory['low'])
#         rule2 = ctrl.Rule(emission['average'], satisfactory['medium'])
#         rule3 = ctrl.Rule(emission['good'] | cost['good'], satisfactory['high'])
#
#     rule1.view()
#     satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
#     satisfactoryping = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)
#     # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
#     # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
#     satisfactoryping.input['cost'] = 6.5
#     satisfactoryping.input['emission'] = 9.8
#
#     # Crunch the numbers
#     satisfactoryping.compute()
#     print(satisfactoryping.output['satisfactory'])
#     satisfactory.view(sim=satisfactoryping)
#     print(2)
#
#
# def fuzzy_HP_one(r):
#     cost_preference, speed_preference, delay_preference, transshipment_preference, emissions_preference = R_info[r]
#     # New Antecedent/Consequent objects hold universe variables and membership
#     # functions
#     cost = ctrl.Antecedent(np.arange(0, 1.6, 0.1), 'cost')
#     # emission = ctrl.Antecedent(np.arange(0, 1.6, 0.1), 'emission')
#     # transshipment = ctrl.Antecedent(np.arange(0, 6, 1), 'transshipment')
#     # delay = ctrl.Antecedent(np.arange(0, 6, 1), 'delay')
#     # speed = ctrl.Antecedent(np.arange(0, 131, 1), 'speed')
#
#     satisfactory = ctrl.Consequent(Satisfaction_universe, 'satisfactory')
#
#     # Auto-membership function population is possible with .automf(3, 5, or 7)
#     # cost
#     cost['Level1'] = fuzz.trapmf(cost.universe, [0, 0, 0.28, 0.43])
#     cost['Level2'] = fuzz.trimf(cost.universe, [0.28, 0.43, 0.75])
#     cost['Level3'] = fuzz.trimf(cost.universe, [0.43, 0.75, 1])
#     cost['Level4'] = fuzz.trapmf(cost.universe, [0.75, 1, 1.5, 1.5])
#     # # emission
#     # emission['Level1'] = fuzz.trapmf(emission.universe, [0, 0, 0.25, 0.33])
#     # emission['Level2'] = fuzz.trimf(emission.universe, [0.25, 0.33, 0.91])
#     # emission['Level3'] = fuzz.trimf(emission.universe, [0.33, 0.91, 1.3])
#     # emission['Level4'] = fuzz.trapmf(emission.universe, [0.91, 1.3, 1.5, 1.5])
#     # # transshipment
#     # transshipment['Level1'] = fuzz.trapmf(transshipment.universe, [0, 0, 1, 2])
#     # transshipment['Level2'] = fuzz.trimf(transshipment.universe, [1, 2, 3])
#     # transshipment['Level3'] = fuzz.trimf(transshipment.universe, [2, 3, 4])
#     # transshipment['Level4'] = fuzz.trapmf(transshipment.universe, [3, 4, 5, 5])
#     # # delay
#     # delay['Level1'] = fuzz.trapmf(delay.universe, [0, 0, 1, 2])
#     # delay['Level2'] = fuzz.trimf(delay.universe, [1, 2, 3])
#     # delay['Level3'] = fuzz.trimf(delay.universe, [2, 3, 4])
#     # delay['Level4'] = fuzz.trapmf(delay.universe, [3, 4, 5, 5])
#     # # speed
#     # speed['Level4'] = fuzz.trapmf(speed.universe, [0, 0, 10, 40])
#     # speed['Level3'] = fuzz.trimf(speed.universe, [10, 40, 70])
#     # speed['Level2'] = fuzz.trimf(speed.universe, [40, 70, 100])
#     # speed['Level1'] = fuzz.trapmf(speed.universe, [70, 100, 130, 130])
#
#     #satisfactory
#     satisfactory['satisfied'] = fuzz.trimf(satisfactory.universe, [0, 100, 100])
#     satisfactory['unsatisfied'] = fuzz.trimf(satisfactory.universe, [0, 0, 100])
#     # satisfactory['high'] = fuzz.trimf(satisfactory.universe, [50, 50, 100])
#
#     # You can see how these look with .view()
#     cost['average'].view()
#     # emission.view()
#     satisfactory.view()
#
#     #the rules are different for different requests
#     #general rule: all preferred levels are met, then high; high preferred levels are met, then medium; only low preferred levels are met, then low
#     def generate_rules(preferred_level, attribute, satisfy):
#         if preferred_level == 1:
#             if satisfy == 1:
#                 return attribute + "['Level1']"
#             else:
#                 return attribute + "['Level2']" + ' | ' + attribute + "['Level3']" + ' | ' + attribute + "['Level4']"
#         elif preferred_level == 2:
#             return attribute + "['Level1']" + ' | ' + attribute + "['Level2']"
#
#
#     rule1 = ctrl.Rule(exec(generate_rules(preferred_level, attribute, 1)), satisfactory['satisfied'])
#     rule2 = ctrl.Rule(exec(generate_rules(preferred_level, attribute, 0)), satisfactory['unsatisfied'])
#     # rule3 = ctrl.Rule(emission['good'] | cost['good'], satisfactory['high'])
#
#     rule1.view()
#     satisfactoryping_ctrl = ctrl.ControlSystem([rule1, rule2])
#     satisfactoryping = ctrl.ControlSystemSimulation(satisfactoryping_ctrl)
#     # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
#     # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
#     satisfactoryping.input['cost'] = 6.5
#     # satisfactoryping.input['emission'] = 9.8
#
#     # Crunch the numbers
#     satisfactoryping.compute()
#     print(satisfactoryping.output['satisfactory'])
#     satisfactory.view(sim=satisfactoryping)
#     print(2)
