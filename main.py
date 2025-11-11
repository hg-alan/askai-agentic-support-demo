import textwrap
from rich.console import Console
from rich.markdown import Markdown

from rag_agent import build_index, answer_question

console = Console()

def run_cli():
    _, chunks = build_index()
    console.print(f"[green]Index built with {chunks} chunks.[/green]\n")

    console.print("[bold]AskAI Agentic Support Demo (CLI)[/bold]")
    console.print("Type your question, or 'exit' to quit.\n")

    while True:
        console.print("[bold magenta]You:[/bold magenta] ", end="")
        question = input().strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            console.print("\n[bold yellow]Goodbye.[/bold yellow]")
            break

        console.print("[dim]Thinking (may decide to escalate)...[/dim]")
        answer, context, meta = answer_question(question)

        console.print("\n[bold cyan]Agent:[/bold cyan]")
        console.print(Markdown(textwrap.dedent(answer)))
        console.print("")

        # Show reasoning summary
        console.print("[bold blue]Agent reasoning:[/bold blue]")
        if meta["mode"] == "escalated":
            console.print(
                f"- Mode: escalation via `{meta['tool_called']}` "
                f"(ticket {meta['ticket']['ticket_id']})"
            )
        else:
            console.print("- Mode: answered directly from retrieved documentation.")
        console.print("")

if __name__ == "__main__":
    run_cli()
