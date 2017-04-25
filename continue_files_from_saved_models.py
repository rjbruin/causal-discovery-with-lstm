from tools.file import load_parameters_with_filename, save_for_continuing;

def create_continue_files_from_saved_model(saved_model_name, iteration):
    filename = "./saved_models/%s_%d.model" % (saved_model_name, iteration);
    # Load params from saved model
    parameters = load_parameters_with_filename(filename);
    # Construct fake variables for .other file
    otherVars = {'val_error_stack': [],
                 'mean_error_stack': [],
                 'last_val_error_avg': 0.};
    # Call original method that creates the right files (to preserve 
    # functionality when the original methods change)
    save_for_continuing(saved_model_name, iteration, 0, 
                        [], otherVars, 
                        parameters,
                        remove=False,
                        saveModel=False);

if __name__ == '__main__':
    # Get saved model filename to work from (without .model)
    name = raw_input("Saved model name (without iteration and extension): ");
    it = int(raw_input("Iteration: "));
    
    create_continue_files_from_saved_model(name, it);