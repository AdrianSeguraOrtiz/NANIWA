"""File-based consensus and the complete inference-to-consensus workflow."""
from .consensus import ConsensusOptions, TemporalConsensus
from .inference import run_inference
from .inputs import load_dataset, load_networks
from .outputs import output_directory, write_consensus


def consensus(dataset, out, *, networks=None, individual=None, expression=None, samples=None,
              regulators=None, options=ConsensusOptions()):
    """Score a joint raw network or individual network files for one dataset."""
    destination = output_directory(out)
    data = load_dataset(dataset, expression, samples, regulators)
    tools, raw = load_networks(data, merged=networks, individual=individual)
    scores = TemporalConsensus(data).score(raw, options=options)
    path = write_consensus(destination, data, scores, tools, raw if individual is not None else None,
                           options=options)
    print(f"{len(tools)} source networks, {data.mask.sum()} eligible interactions → {path}")
    return path


def run(dataset, out, *, options=ConsensusOptions(), **inference_options):
    """Infer once and pass ANDREA's frozen data and raw export directly to consensus."""
    root = output_directory(out)
    result = run_inference(dataset, root / "inference", **inference_options)
    print(result, flush=True)
    return consensus(result.dataset, root / "consensus", networks=result.network, options=options)
