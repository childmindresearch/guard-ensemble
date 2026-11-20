import json

import polars as pl

import guard_ensemble.domains.model as domains
import guard_ensemble.domains.post_process as pproc
import guard_ensemble.guards.model as guards
from guard_ensemble import dispatch, util
from guard_ensemble.llm import catalog, runner, worker
from guard_ensemble.sources import mirror


def reload_modules():
    import importlib

    importlib.reload(catalog)
    importlib.reload(domains)
    importlib.reload(pproc)
    importlib.reload(guards)
    importlib.reload(util)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(mirror)
    importlib.reload(dispatch)


runner.stop()

reload_modules()

## Benchmark Data --------------------------------------------------------------

labelled_data, _ = util.load_input_csv("data/labelled_clinician_inputs.csv")

llm_mock_data, _ = util.load_input_csv("data/sample_inputs.csv")

res, _ = domains.run(labelled_data)

# res = domains.run(llm_mock_data)

util.save_out_jsonl(res, "domains_labelled_0001")

## Mirror ----------------------------------------------------------------------

records = mirror.pull(limit=1000)

mirror_df = mirror.select_msm_frame(mirror.frame_records(records))

##

dispatch.run_all(mirror_df, util.PACKAGE_DIR / "out" / "dispatch")

##

mres, _ = domains.run(mirror_df)

m_unsafe = mres.filter((pl.col("concerning") == True) | (pl.col("high_risk") == True))
d_unsafe = mres.remove(entry_tier="green")

d_missed = m_unsafe.filter(
    pl.col("entry_tier").is_null() | (pl.col("entry_tier") == "green")
)

d_unsafe["domain_results"][0]

drs = pl.col(d_unsafe["domain_results"].struct.fields)

mres["domain_raw"]

j = (
    d_unsafe
    .unnest("domain_results")
    .filter(pl.any_horizontal(drs.struct.field("tier") != "green"))
    .with_columns(drs.struct.field("tier").name.keep())
    .select("text", drs)
    .write_json()
)

print(json.dumps(j))

print(d_missed["text"][1])

print(f"{m_unsafe.filter(pl.col('entry_tier').is_null())}")


###
