"""'Hub' code, to run experiments."""
import time


def run_experiment(data_getter, model_maker, evaluator, modifier):
    """Run the experiment.  This consists of getting the data, creating
    the model (including training) and evaluating the results.

    """
    train_data, test_data = data_getter()
    model = model_maker(train_data)
    model = modifier(model)
    if isinstance(test_data, tuple):
        start = time.clock()
        eval_results = evaluator(model, test_data)
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
