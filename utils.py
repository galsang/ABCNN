
def build_path(prefix, data_type, model_type, num_layers, postpix=""):
    return prefix + data_type + "-" + model_type + "-" + str(num_layers) + postpix
