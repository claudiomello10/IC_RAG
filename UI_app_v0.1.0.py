import os
import warnings
import customtkinter as ctk
from datetime import datetime

# Remove warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Import the LLM_IC class
from LLM_IC import LLM_IC


class ScrollableChatFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.message_widgets = []
        self.streaming_label = None

    def add_message(self, message: str, is_user: bool = False):
        # Create message frame
        message_frame = ctk.CTkFrame(self, fg_color="transparent")
        message_frame.grid(
            row=len(self.message_widgets), column=0, padx=10, pady=5, sticky="ew"
        )
        message_frame.grid_columnconfigure(0, weight=1)

        # Time stamp
        time_label = ctk.CTkLabel(
            message_frame,
            text=datetime.now().strftime("%H:%M"),
            font=("Helvetica", 10),
            text_color="gray",
        )

        # Message bubble
        bubble_frame = ctk.CTkFrame(
            message_frame,
            fg_color="#2C88D9" if is_user else "#404040",
            corner_radius=15,
        )

        message_label = ctk.CTkLabel(
            bubble_frame,
            text=message,
            font=("Helvetica", 12),
            wraplength=400,
            justify="left" if not is_user else "right",
        )
        message_label.pack(padx=12, pady=8)

        # Layout based on sender
        if is_user:
            bubble_frame.grid(row=0, column=0, sticky="e")
            time_label.grid(row=1, column=0, sticky="e", padx=5)
        else:
            bubble_frame.grid(row=0, column=0, sticky="w")
            time_label.grid(row=1, column=0, sticky="w", padx=5)

        self.message_widgets.append(message_frame)
        self._parent_canvas.yview_moveto(1.0)  # Scroll to bottom

        if not is_user:
            self.streaming_label = message_label

    def update_last_message(self, message: str):
        if self.streaming_label:
            self.streaming_label.configure(text=message)
            self._parent_canvas.yview_moveto(1.0)


class ModernChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("IC Chatbot")
        self.geometry("800x600")

        # Configure the grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create main layout
        self.create_sidebar()
        self.create_main_chat()

        # Initialize LLM
        self.llm = LLM_IC(
            embeddings_paths=[
                "full_df_embeddings_sections1.json",
                "full_df_embeddings_sections2.json",
            ]
        )

        # Start with system theme
        ctk.set_appearance_mode("system")

    def create_sidebar(self):
        # Create sidebar frame
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_rowconfigure(4, weight=1)

        # App logo/title
        logo_label = ctk.CTkLabel(
            sidebar, text="IC Chatbot", font=ctk.CTkFont(size=20, weight="bold")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Theme switch
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["System", "Light", "Dark"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode_menu.grid(row=1, column=0, padx=20, pady=10)

        # Clear chat button
        clear_button = ctk.CTkButton(
            sidebar, text="Clear Chat", command=self.clear_chat
        )
        clear_button.grid(row=2, column=0, padx=20, pady=10)

        # Version info at bottom
        version_label = ctk.CTkLabel(sidebar, text="v0.1.0", text_color="gray")
        version_label.grid(row=5, column=0, padx=20, pady=(10, 20))

    def create_main_chat(self):
        # Create main chat frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create chat display
        self.chat_display = ScrollableChatFrame(main_frame, width=200, corner_radius=10)
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        # Create input frame
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)

        # Create input field
        self.input_field = ctk.CTkTextbox(input_frame, height=40, corner_radius=10)
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        # Create send button
        self.send_button = ctk.CTkButton(
            input_frame, text="Send", width=100, command=self.send_message
        )
        self.send_button.grid(row=0, column=1)

        # Bind enter key to send message
        self.input_field.bind("<Return>", self.handle_return)

    def handle_return(self, event):
        if not (event.state & 0x1):  # Shift not pressed
            self.send_message()
            return "break"

    def send_message(self):
        message = self.input_field.get("1.0", "end-1c").strip()
        if message:
            # Clear input field again to ensure it's empty
            self.input_field.delete("1.0", "end")
            # Disable input temporarily
            self.input_field.configure(state="disabled")
            self.send_button.configure(state="disabled")

            # Clear input field
            self.input_field.delete("1.0", "end")

            # Show user message
            self.chat_display.add_message(message, is_user=True)

            # Start AI response
            self.chat_display.add_message("")  # Add empty message for streaming
            ai_response = ""

            # Get AI response
            try:
                response = self.llm.generate_response_stream(message)
                for r in response:
                    if r.choices[0].delta.content is not None:
                        ai_response += r.choices[0].delta.content
                        self.chat_display.update_last_message(ai_response)
                        self.update()
            except Exception as e:
                self.chat_display.update_last_message(f"Error: {str(e)}")
            finally:
                # Re-enable input
                self.input_field.configure(state="normal")
                self.send_button.configure(state="normal")
                self.input_field.focus()

    def clear_chat(self):
        for widget in self.chat_display.message_widgets:
            widget.destroy()
        self.chat_display.message_widgets.clear()
        self.chat_display.streaming_label = None

    def change_appearance_mode(self, mode: str):
        ctk.set_appearance_mode(mode.lower())


if __name__ == "__main__":
    app = ModernChatApp()
    app.mainloop()
