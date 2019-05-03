import time

def run_experiment(get_data, get_model, modifier):
    """Run the experiment.  This consists of getting the data, creating
    the model (including training) and evaluating the results.

    """
    train_data, test_data = get_data()
    model = get_model(train_data)
    model = modifier(model)
    if isinstance(test_data, tuple):
        start = time.clock() 
        eval_results = model.evaluate(*test_data)
        end = time.clock()
        result = {
            'loss': eval_results[0],
            'acc': eval_results[1],
            'time': end - start
        }
    else:
        msg = 'The test data is of type which can not be' + \
            ' handeled by the current implementation.'
        raise NotImplementedError(msg)
    return result

