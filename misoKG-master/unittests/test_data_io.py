from data_io import match_data_filename, check_file_legitimate

def tests_match_data_filename_rb():
    for func_name in ["rbCvanN", "rbCsinN", "rbCbiasN"]:
        for idx in [1, 3, 5, 9]:
            yield check_match_data_filename, func_name, idx, 25

def tests_match_data_filename_ato():
    for func_name in ["atoC_vanilla", "atoC_var2", "atoC_var3", "atoC_var4"]:
        for idx in [1, 3, 5, 9]:
            yield check_match_data_filename, func_name, idx, 50

def check_match_data_filename(func_name, idx, num_data):
    print match_data_filename("/fs/europa/g_pf/pickles/coldstart/data", "seqpes_{0}_repl".format(func_name), idx, check_file_legitimate, num_data=num_data)