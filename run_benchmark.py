from __future__ import annotations
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = typer.Option("openai", help="Runtime mode: openai or mock."),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model name when mode=openai."),
) -> None:
    load_dotenv()
    runtime_mode = mode.strip().lower()
    if runtime_mode not in {"mock", "openai"}:
        raise typer.BadParameter("mode must be either 'mock' or 'openai'.")
    if runtime_mode == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
        raise typer.BadParameter("OPENAI_API_KEY is not set. Configure it in environment or .env before running openai mode.")

    examples = load_dataset(dataset)
    react = ReActAgent(mode=runtime_mode, model=model)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode=runtime_mode, model=model)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=runtime_mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
