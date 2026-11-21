import time
import random
from rich.console import Console
from rich.progress import track
from rich.panel import Panel  # 'pane' ko 'panel' kiya

console = Console()

console.print(Panel.fit(
    "[bold cyan]Tech by Niteshh // Network Diagnostic engine v1.0[/bold cyan]",
    title="[bold green]Network Diagnostic Engine[/bold green]",
    border_style="bright_blue"
))
time.sleep(1)

targets = [
    "192.168.1.1 (Gateway)",
    "10.0.0.42",
    "165.256.25.2 (IOT node)",
    "586.25.25.1 (cloud - relay)"
]

for step in track(range(25), description="[bold yellow]Scanning network...[/bold yellow]"):
    time.sleep(random.uniform(0.1, 0.3))

console.print(Panel.fit(
    "[bold green]Network scan complete![/bold green]",
    title="[bold blue]Scan Results[/bold blue]",
    border_style="bright_blue"
))

console.print("\n[bold white][*] Active nodes discovered:[/bold white]")

for target in targets:
    latency = random.randint(4, 28)
    # String ko ek hi line mein close kiya aur har target ke beech thoda pause add kiya
    console.print(f"[bold cyan]{target}[/bold cyan] - Latency: [bold yellow]{latency}ms[/bold yellow]")
    time.sleep(0.3)

console.print("\n")
console.print(Panel.fit(
    "[bold green]Network diagnostic completed successfully![/bold green]",
    title="[bold blue]Summary[/bold blue]",
    border_style="bright_blue"
))