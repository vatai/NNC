"""'Hub' code, to run experiments."""
import time


def run_experiment(data_getter, model_maker, trainer=None,
                   evaluator=None, modifier=None):
    """Run the experiment.  This consists of getting the data, creating
    the model (maybe training) and evaluating the results.

    """
    train_data, test_data = data_getter()
    model = model_maker()

    if modifier is not None:
        model = modifier(model)

    if trainer is not None:
        trainer(model, train_data)

    start = time.clock()
    eval_results = evaluator(model, test_data)
    end = time.clock()

    result = {
        'loss': eval_results[0],
        'acc': eval_results[1],
        'time': end - start
    }
    return result
