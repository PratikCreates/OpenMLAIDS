import json
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Input, RichLog, LoadingIndicator, DataTable, TabbedContent, TabPane, Button, Static, Checkbox

from textual.screen import ModalScreen
from textual import on, work

# IMPORT THE BRAIN (This loads agent.py)
# IMPORT THE BRAIN (This loads agent.py)
from agent import app as agent_app
from langchain_core.messages import ToolMessage


AUTO_APPROVE_TOOLS = ["web_search", "inspect_data", "python_helper", "download_kaggle_dataset"]

CSS = """
Screen { layout: horizontal; }
#sidebar {
    width: 25;
    dock: left;
    background: #1e1e1e;
    border-right: solid #333;
    padding: 1;
    color: #888;
}
#main-area { width: 1fr; }
#chat-log {
    height: 1fr;
    border: solid #333;
    background: #0d1117;
    padding: 1 1;
}
Input {
    dock: bottom;
    height: 3;
    border: solid #2ea043;
}
LoadingIndicator {
    height: 1;
    dock: top;
    display: none;
    color: #2ea043;
}
.visible { display: block; }
DataTable {
    height: 1fr;
    border: solid #444;
}

/* Modal Styles */
ToolApprovalModal {
    align: center middle;
}
#modal-container {
    width: 60;
    height: auto;
    max-height: 80%; /* Allow more height but keep it reasonable */
    background: #1e1e1e;
    border: solid #ff6600;
    padding: 1 2;
}
#modal-tool-info {
    padding: 1;
    background: #0d1117;
    margin-bottom: 1;
    max-height: 20; /* Limit text area height */
    overflow-y: auto; /* Allow scrolling if text is long */
}
#modal-buttons {
    align: center middle;
    height: 3;
    dock: bottom; /* Ensure buttons stay at bottom if we used a container */
}
#modal-buttons Button {
    margin: 0 1;
}
#approve-btn {
    background: #2ea043;
}
#deny-btn {
    background: #da3633;
}
"""


class ToolApprovalModal(ModalScreen[bool]):
    """Modal screen for tool approval."""
    
    def __init__(self, tool_name: str, tool_args: dict, tool_call_id: str):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_id = tool_call_id
    
    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Static("⚠️ Tool Approval Required", id="modal-title")
            
            # CUSTOM DISPLAY LOGIC
            display_text = ""
            if self.tool_name == "python_helper":
                display_text = f"[bold cyan]{self.tool_name}[/]\n[italic]Requesting to execute Python code...[/]"
                # Optional: Show a preview line if possible, or just keep it simple as requested
            else:
                # Default for other tools
                display_text = f"[bold cyan]{self.tool_name}[/]\n[dim]{self.tool_args}[/]"
            
            yield Static(display_text, id="modal-tool-info")
            
            with Horizontal(id="modal-buttons"):
                yield Button("✓ Approve", id="approve-btn", variant="success")
                yield Button("✗ Deny", id="deny-btn", variant="error")
    
    @on(Button.Pressed, "#approve-btn")
    def approve(self) -> None:
        self.dismiss(True)
    
    @on(Button.Pressed, "#deny-btn")
    def deny(self) -> None:
        self.dismiss(False)


class OpenMLAIDSApp(App):
    CSS = CSS
    BINDINGS = [("q", "quit", "Quit")]
    
    def __init__(self):
        super().__init__()
        self.current_config = None
        self.pending_tool_calls = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="sidebar"):
            yield RichLog(id="status-log", markup=True)
            yield Checkbox("Safe Mode (Auto-Approve)", True, id="chk-auto-approve")
            
        with Vertical(id="main-area"):
            yield LoadingIndicator(id="spinner")
            
            # TABS
            with TabbedContent(initial="tab-chat"):
                with TabPane("Chat", id="tab-chat"):
                    yield RichLog(id="chat-log", highlight=True, markup=True)
                
                with TabPane("Data Inspector", id="tab-data"):
                    yield DataTable(id="data-table")
            
            yield Input(placeholder="Ask OpenMLAIDS...", id="input-box")
            
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-log").write("[bold green]=== System Online (Azure GPT-5.2) ===[/]")
        self.query_one("#status-log").write("[bold]Active Session:[/]\nSession_1")

    @on(Input.Submitted)
    def on_user_input(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text: return
        self.query_one("#input-box").value = ""
        self.query_one("#chat-log").write(f"\n[bold yellow]User >[/] {user_text}")
        self.query_one("#spinner").add_class("visible")
        self.run_agent_loop(user_text)

    @work(exclusive=True, thread=True)
    def run_agent_loop(self, user_input: str):
        config = {"configurable": {"thread_id": "session_1"}}
        self.current_config = config
        input_message = {"messages": [("user", user_input)]}
        
        try:
            # Stream until we hit an interrupt (tool call)
            for event in agent_app.stream(input_message, config=config):
                self.app.call_from_thread(self.process_agent_event, event)
            
            # Check if there's a pending interrupt (tool calls waiting for approval)
            state = agent_app.get_state(config)
            if state.next:  # There are pending nodes (tools waiting)
                # Get the last message to find tool calls
                last_msg = state.values["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if self.should_auto_approve(tc['name']):
                             self.app.call_from_thread(self.notify_auto_approve, tc['name'])
                             self.continue_agent_after_approval()
                             return

                        self.app.call_from_thread(
                            self.request_tool_approval, 
                            tc['name'], 
                            tc['args'],
                            tc['id']
                        )
                        return  # Wait for approval
                        
        except Exception as e:
            self.app.call_from_thread(self.log_error, str(e))
        
        self.app.call_from_thread(self.hide_spinner)

    def request_tool_approval(self, tool_name: str, tool_args: dict, tool_call_id: str):
        """Show modal to request tool approval."""
        def handle_approval(approved: bool) -> None:
            if approved:
                self.query_one("#chat-log").write(f"[bold green]✓ Approved:[/] {tool_name}")
                self.continue_agent_after_approval()
            else:
                self.query_one("#chat-log").write(f"[bold red]✗ Denied:[/] {tool_name}")
                self.handle_denial(tool_call_id)
        
        self.push_screen(ToolApprovalModal(tool_name, tool_args, tool_call_id), handle_approval)
    
    @work(exclusive=True, thread=True)
    def handle_denial(self, tool_call_id: str):
        """Inject a rejection message so the agent knows it was denied and state remains valid."""
        try:
            # Create a ToolMessage with error-like content
            rejection_msg = ToolMessage(
                tool_call_id=tool_call_id,
                content="User denied execution. Do not retry the same exact command unless asked."
            )
            # Update state AS IF the tool ran
            agent_app.update_state(self.current_config, {"messages": [rejection_msg]}, as_node="tools")
            
            # Resume stream to let agent react (it will see the denial)
            for event in agent_app.stream(None, config=self.current_config):
                self.app.call_from_thread(self.process_agent_event, event)
                
        except Exception as e:
            self.app.call_from_thread(self.log_error, str(e))
        
        self.app.call_from_thread(self.hide_spinner)

    @work(exclusive=True, thread=True)
    def continue_agent_after_approval(self):
        """Continue the agent execution after tool approval."""
        try:
            # Continue with None input to resume after interrupt
            for event in agent_app.stream(None, config=self.current_config):
                self.app.call_from_thread(self.process_agent_event, event)
            
            # Check if there are more tools waiting
            state = agent_app.get_state(self.current_config)
            if state.next:
                last_msg = state.values["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if self.should_auto_approve(tc['name']):
                             self.app.call_from_thread(self.notify_auto_approve, tc['name'])
                             self.continue_agent_after_approval()
                             return

                        self.app.call_from_thread(
                            self.request_tool_approval,
                            tc['name'],
                            tc['args'],
                            tc['id']
                        )
                        return
                        
        except Exception as e:
            self.app.call_from_thread(self.log_error, str(e))
        
        self.app.call_from_thread(self.hide_spinner)

    def process_agent_event(self, event):
        chat = self.query_one("#chat-log")
        
        for key, value in event.items():
            
            # 1. THE AGENT SPEAKS (OR CALLS A TOOL)
            if key == "chatbot":
                msg = value['messages'][-1]
                
                # If the agent wants to call tools, show them clearly!
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc['name']
                        tool_args = tc['args']
                        chat.write(f"[bold yellow]⚡ Tool Request:[/] [cyan]{tool_name}[/] [dim]{tool_args}[/]")
                
                # If the agent has a text response, print it
                if msg.content:
                    chat.write(f"\n[bold cyan]Agent >[/] {msg.content}")
            
            # 2. THE TOOL RESPONDS
            elif key == "tools":
                msg = value['messages'][-1]
                tool_name = msg.name
                
                # Special Case: Data Inspection (Load Table, don't spam log)
                if tool_name == "inspect_data":
                    try:
                        data = json.loads(msg.content)
                        if "error" in data:
                            chat.write(f"[bold red]❌ Tool Error:[/] {data['error']}")
                        else:
                            self.load_data_to_table(data)
                            chat.write(f"[italic green]✔ Data Loaded: {len(data['data'])} rows sent to Inspector Tab[/]")
                    except:
                        chat.write(f"[red]❌ Error parsing JSON from inspect_data[/]")
                
                # General Case: Shell / Download / Etc
                else:
                    # Print the actual output from the tool
                    output_text = msg.content
                    # Truncate if massive (optional)
                    if len(output_text) > 500:
                        output_text = output_text[:500] + "... (truncated)"
                    
                    chat.write(f"[dim]  ↳ Result: {output_text}[/]")
                    
                    # 3. CHECK FOR IMAGES
                    if ".png" in output_text or ".jpg" in output_text:
                         chat.write(f"[bold magenta]🖼️ Image generated! Check your workspace folder.[/]")


    def load_data_to_table(self, data_dict):
        table = self.query_one("#data-table", DataTable)
        table.clear(columns=True)
        table.add_columns(*data_dict['columns'])
        table.add_rows(data_dict['data'])
        self.query_one(TabbedContent).active = "tab-data"

    def hide_spinner(self):
        self.query_one("#spinner").remove_class("visible")
    def log_error(self, m):
        self.query_one("#chat-log").write(f"[red]Error: {m}[/]")

    def should_auto_approve(self, tool_name: str) -> bool:
        """Check if tool should be auto-approved based on checkbox and list."""
        # Checkbox logic: If "Safe Mode" is CHECKED, we auto-approve known tools.
        # If UNCHECKED, we require approval for everything (Paranoid Mode).
        is_safe_mode = self.query_one("#chk-auto-approve").value
        if not is_safe_mode:
            return False
        return tool_name in AUTO_APPROVE_TOOLS

    def notify_auto_approve(self, tool_name: str):
        self.query_one("#chat-log").write(f"[dim green]⚡ Auto-Approved:[/] {tool_name}")

if __name__ == "__main__":
    app = OpenMLAIDSApp()
    app.run()
