"""
stream_browser.py — Interactive Rich-based stream browser

Displays a table of IPTV streams with search, filter, probe, and select.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

MAX_DISPLAY_ROWS = 40


def run_browser(streams, keyword=None):
    """
    Interactive stream browser loop.

    Args:
        streams: list of Stream objects from stream_discovery
        keyword: optional initial search filter

    Returns:
        str or None: selected stream URL, or None if user quits
    """
    if not streams:
        console.print("[red]No streams found.[/red]")
        return None

    filtered = streams
    if keyword:
        filtered = _search(streams, keyword)

    _print_help()

    while True:
        _display_table(filtered)

        try:
            raw = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            return None

        if not raw:
            continue

        # Quit
        if raw.lower() in ("q", "quit", "exit"):
            return None

        # Search
        if raw.lower().startswith("s "):
            keyword = raw[2:].strip()
            if keyword:
                filtered = _search(streams, keyword)
                console.print(
                    f"[dim]Showing {len(filtered)} results for "
                    f"'{keyword}'[/dim]"
                )
            continue

        # Show all
        if raw.lower() in ("a", "all"):
            filtered = streams
            keyword = None
            continue

        # Probe
        if raw.lower() in ("p", "probe"):
            console.print("[dim]Probing stream availability...[/dim]")
            from stream_discovery import probe_streams_batch
            probe_streams_batch(filtered[:MAX_DISPLAY_ROWS])
            continue

        # Help
        if raw.lower() in ("h", "help", "?"):
            _print_help()
            continue

        # Number selection
        try:
            idx = int(raw)
            if 1 <= idx <= len(filtered):
                selected = filtered[idx - 1]
                result = _confirm_selection(selected)
                if result == "select":
                    return selected.url
                elif result == "quit":
                    return None
                # else "back" — continue loop
            else:
                console.print(f"[red]Invalid number. Enter 1-{len(filtered)}[/red]")
        except ValueError:
            console.print(
                "[red]Unknown command. Type 'h' for help.[/red]"
            )


def _display_table(streams):
    """Render the stream table."""
    table = Table(
        title="IPTV Sports Streams",
        show_lines=False,
        pad_edge=True,
        expand=True,
    )

    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Name", style="bold", max_width=40, no_wrap=True)
    table.add_column("Category", max_width=20, no_wrap=True)
    table.add_column("Country", max_width=8, no_wrap=True)
    table.add_column("Quality", width=6, justify="center")
    table.add_column("Status", width=8, justify="center")
    table.add_column("Type", width=6, justify="center")

    display = streams[:MAX_DISPLAY_ROWS]

    for i, s in enumerate(display, 1):
        # Status styling
        if s.status == "online":
            status = Text("LIVE", style="bold green")
        elif s.status == "offline":
            status = Text("OFF", style="red")
        elif s.status == "timeout":
            status = Text("SLOW", style="yellow")
        else:
            status = Text("-", style="dim")

        # Quality styling
        quality = s.quality or "-"
        if quality in ("4K", "FHD"):
            quality_text = Text(quality, style="bold green")
        elif quality == "HD":
            quality_text = Text(quality, style="cyan")
        else:
            quality_text = Text(quality, style="dim")

        # Type
        stream_type = Text("ACE", style="magenta") if s.is_acestream else Text("HTTP", style="dim")

        # Alternating row style
        row_style = "on grey11" if i % 2 == 0 else ""

        table.add_row(
            str(i), s.name, s.group, s.tvg_country,
            quality_text, status, stream_type,
            style=row_style,
        )

    console.print(table)

    total = len(streams)
    shown = len(display)
    if total > shown:
        console.print(
            f"[dim]Showing {shown}/{total} streams. "
            f"Use 's <keyword>' to search/filter.[/dim]"
        )


def _confirm_selection(stream):
    """
    Show stream details and ask for confirmation.

    Returns:
        'select', 'back', or 'quit'
    """
    info = (
        f"[bold]{stream.name}[/bold]\n"
        f"URL: [cyan]{stream.url}[/cyan]\n"
        f"Group: {stream.group or '-'}\n"
        f"Country: {stream.tvg_country or '-'}\n"
        f"Quality: {stream.quality or 'Unknown'}\n"
        f"Type: {'Ace Stream' if stream.is_acestream else 'HTTP/HLS'}\n"
        f"Status: {stream.status}"
    )
    console.print(Panel(info, title="Selected Stream", border_style="cyan"))

    try:
        choice = console.input(
            "[bold]Watch this stream? [y]es / [n]o / [q]uit: [/bold]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "quit"

    if choice in ("y", "yes", ""):
        return "select"
    elif choice in ("q", "quit"):
        return "quit"
    return "back"


def _search(streams, keyword):
    """Filter streams by keyword match on name, group, or tvg_name."""
    kw = keyword.lower()
    return [
        s for s in streams
        if kw in (s.name + " " + s.group + " " + s.tvg_name + " " + s.tvg_country).lower()
    ]


def _print_help():
    """Print browser commands."""
    help_text = (
        "[bold]Commands:[/bold]\n"
        "  [cyan]<number>[/cyan]    Select stream by number\n"
        "  [cyan]s <word>[/cyan]    Search/filter streams\n"
        "  [cyan]a[/cyan]           Show all streams\n"
        "  [cyan]p[/cyan]           Probe stream availability\n"
        "  [cyan]h[/cyan]           Show this help\n"
        "  [cyan]q[/cyan]           Quit"
    )
    console.print(Panel(help_text, title="Stream Browser", border_style="dim"))
