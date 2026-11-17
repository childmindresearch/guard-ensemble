import guard_ensemble.domains.model as domains
from guard_ensemble.guards import lib
from guard_ensemble.llm import catalog, runner, worker


def reload_modules():
    import importlib

    importlib.reload(domains)
    importlib.reload(lib)
    importlib.reload(catalog)
    importlib.reload(worker)
    importlib.reload(runner)


runner.stop()

reload_modules()

labelled_data = lib.load_input_csv("data/labelled_clinician_inputs.csv")
llm_mock_data = lib.load_input_csv("data/sample_inputs.csv")

res = domains.run(labelled_data)
# res = domains.run(llm_mock_data)

lib.save_out_jsonl(res, "key_domains_qwen3_4b_2507_labelled_0005")
