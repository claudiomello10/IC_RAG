import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import json
from datetime import datetime
from MultiPurpuse_LLM import StudentHelper


class BubbleMessage(ctk.CTkFrame):
    def __init__(self, master, message, timestamp, is_user=False, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Configure grid
        self.grid_columnconfigure(1, weight=1)

        # Create bubble frame with different colors for user/assistant
        bubble_frame = ctk.CTkFrame(
            self, fg_color="#2C88D9" if is_user else "#404040", corner_radius=15
        )

        # Create message label
        message_label = ctk.CTkLabel(
            bubble_frame,
            text=message,
            text_color="white",
            wraplength=400,
            justify="left",
            anchor="w",
        )
        message_label.pack(padx=12, pady=8)

        # Create timestamp label
        time_label = ctk.CTkLabel(
            self, text=timestamp, font=("Helvetica", 10), text_color="gray"
        )

        # Position elements based on sender
        if is_user:
            bubble_frame.grid(row=0, column=1, sticky="e", padx=(50, 10))
            time_label.grid(row=1, column=1, sticky="e", padx=5)
        else:
            bubble_frame.grid(row=0, column=0, sticky="w", padx=(10, 50))
            time_label.grid(row=1, column=0, sticky="w", padx=5)


class ScrollableChatFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.messages = []
        self.last_assistant_bubble = None

    def add_message(self, message, is_user=False):
        timestamp = datetime.now().strftime("%H:%M")
        bubble = BubbleMessage(self, message, timestamp, is_user)
        bubble.grid(row=len(self.messages), column=0, sticky="ew", pady=5)
        self.messages.append(bubble)

        if not is_user:
            self.last_assistant_bubble = bubble

        self._parent_canvas.yview_moveto(1.0)

    def update_last_assistant_message(self, message):
        if self.last_assistant_bubble:
            for widget in self.last_assistant_bubble.winfo_children():
                if isinstance(widget, ctk.CTkFrame):  # This is the bubble frame
                    for label in widget.winfo_children():
                        if isinstance(label, ctk.CTkLabel):  # This is the message label
                            label.configure(text=message)
                            break
            self._parent_canvas.yview_moveto(1.0)


class StudentHelperGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Student Helper")
        self.geometry("1200x800")

        # Initialize StudentHelper
        try:
            self.student_helper = StudentHelper()
            self.has_embeddings = False
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to initialize StudentHelper: {str(e)}"
            )
            return

        # Store button references
        self.buttons = {}

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create frames
        self.create_sidebar()
        self.create_main_area()

    def create_sidebar(self):
        # Create sidebar frame
        sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)

        # App title with icon
        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(pady=20, padx=10, fill="x")

        title = ctk.CTkLabel(
            title_frame, text="Student Helper", font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack()

        # Subject Selection Section
        subject_frame = ctk.CTkFrame(sidebar)
        subject_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            subject_frame, text="Subject", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(5, 0))

        self.subject_entry = ctk.CTkEntry(subject_frame, height=35)
        self.subject_entry.pack(pady=5, padx=10, fill="x")

        self.buttons["set_subject"] = ctk.CTkButton(
            subject_frame, text="Set Subject", command=self.set_subject, height=35
        )
        self.buttons["set_subject"].pack(pady=5, padx=10, fill="x")

        # PDF Management Section
        pdf_frame = ctk.CTkFrame(sidebar)
        pdf_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            pdf_frame, text="PDF Management", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(5, 0))

        self.buttons["load_pdf"] = ctk.CTkButton(
            pdf_frame, text="Load PDF", command=self.load_pdf, height=35
        )
        self.buttons["load_pdf"].pack(pady=5, padx=10, fill="x")

        # Embeddings Management Section
        emb_frame = ctk.CTkFrame(sidebar)
        emb_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            emb_frame, text="Embeddings", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(5, 0))

        self.buttons["load_embeddings"] = ctk.CTkButton(
            emb_frame, text="Load Embeddings", command=self.load_embeddings, height=35
        )
        self.buttons["load_embeddings"].pack(pady=5, padx=10, fill="x")

        self.buttons["save_embeddings"] = ctk.CTkButton(
            emb_frame, text="Save Embeddings", command=self.save_embeddings, height=35
        )
        self.buttons["save_embeddings"].pack(pady=5, padx=10, fill="x")

        # Settings Section
        settings_frame = ctk.CTkFrame(sidebar)
        settings_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            settings_frame, text="Settings", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(5, 0))

        # Model selection
        self.model_var = ctk.StringVar(value="gpt-4o-mini")
        self.model_dropdown = ctk.CTkOptionMenu(
            settings_frame,
            values=["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"],
            variable=self.model_var,
            command=self.change_model,
            height=35,
        )
        self.model_dropdown.pack(pady=5, padx=10, fill="x")

        # Status display at bottom of sidebar
        self.status_label = ctk.CTkLabel(
            sidebar,
            text="Ready",
            font=ctk.CTkFont(size=12),
            fg_color=("#EEEEEE", "#333333"),
            corner_radius=8,
        )
        self.status_label.pack(side="bottom", pady=10, padx=10, fill="x")

    def create_main_area(self):
        # Create main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Chat display
        self.chat_display = ScrollableChatFrame(main_frame)
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))

        # Input area
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)

        # Create input box
        self.input_field = ctk.CTkTextbox(input_frame, height=100, corner_radius=10)
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_field.bind("<Return>", self.handle_return)

        # Send button
        self.buttons["send"] = ctk.CTkButton(
            input_frame, text="Send", width=100, height=40, command=self.send_message
        )
        self.buttons["send"].grid(row=0, column=1)

    def set_subject(self):
        subject = self.subject_entry.get().strip()
        if not subject:
            messagebox.showerror("Error", "Please enter a subject!")
            return

        try:
            self.student_helper.subject = subject
            self.update_status(f"Subject set to: {subject}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set subject: {str(e)}")

    def toggle_buttons(self, enabled=True):
        for button in self.buttons.values():
            button.configure(state="normal" if enabled else "disabled")
        self.model_dropdown.configure(state="normal" if enabled else "disabled")
        self.subject_entry.configure(state="normal" if enabled else "disabled")
        self.input_field.configure(state="normal" if enabled else "disabled")

    def handle_return(self, event):
        if not (event.state & 0x1):  # Shift not pressed
            self.send_message()
            return "break"

    def send_message(self):
        if not self.has_embeddings:
            messagebox.showerror("Error", "Please load PDFs or embeddings first!")
            return

        message = self.input_field.get("1.0", "end-1c").strip()
        if not message:
            return

        self.input_field.delete("1.0", "end")
        self.chat_display.add_message(message, is_user=True)

        # Disable buttons during response generation
        self.toggle_buttons(False)

        def generate_response():
            try:
                # Add empty assistant message first
                self.chat_display.add_message("Thinking...")

                # Generate response
                response = self.student_helper.generate_response(message)
                self.chat_display.update_last_assistant_message(response)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                self.chat_display.update_last_assistant_message(error_msg)
                messagebox.showerror("Error", error_msg)
            finally:
                self.toggle_buttons(True)
                self.input_field.focus()

        threading.Thread(target=generate_response, daemon=True).start()

    def update_status(self, message):
        self.status_label.configure(text=message)
        self.update()

    def add_message_to_display(self, message, is_user=False):
        self.chat_display.add_message(message, is_user)

    def load_pdf(self):
        files = filedialog.askopenfilenames(
            title="Select PDF files", filetypes=[("PDF files", "*.pdf")]
        )

        if files:

            def process_pdfs():
                self.toggle_buttons(False)
                try:
                    self.update_status("Processing PDFs...")
                    book_names = [
                        os.path.splitext(os.path.basename(f))[0] for f in files
                    ]

                    if not hasattr(self.student_helper, "embedding_df"):
                        # First PDF - set embeddings
                        self.student_helper.set_embeddings_df_from_multiple_PDFs(
                            files, book_names, chunk_size=3000
                        )
                    else:
                        # Add to existing embeddings
                        self.student_helper.add_multiple_PDFs_to_embeddings_df(
                            files, book_names, chunk_size=3000
                        )

                    self.has_embeddings = True
                    self.update_status("PDFs processed successfully!")
                except Exception as e:
                    self.update_status(f"Error processing PDFs: {str(e)}")
                    messagebox.showerror("Error", f"Failed to process PDFs: {str(e)}")
                finally:
                    self.toggle_buttons(True)

            # Run in separate thread to keep UI responsive
            threading.Thread(target=process_pdfs, daemon=True).start()

    def load_embeddings(self):
        file = filedialog.askopenfilename(
            title="Select embeddings file", filetypes=[("JSON files", "*.json")]
        )

        if file:
            self.toggle_buttons(False)
            try:
                self.update_status("Loading embeddings...")
                self.student_helper.load_embeddings(file)
                self.student_helper.create_faiss_index()
                self.has_embeddings = True
                self.update_status("Embeddings loaded successfully!")
            except Exception as e:
                self.update_status(f"Error loading embeddings: {str(e)}")
                messagebox.showerror("Error", f"Failed to load embeddings: {str(e)}")
            finally:
                self.toggle_buttons(True)

    def save_embeddings(self):
        if not hasattr(self.student_helper, "embedding_df"):
            messagebox.showerror("Error", "No embeddings to save!")
            return

        file = filedialog.asksaveasfilename(
            title="Save embeddings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )

        if file:
            self.toggle_buttons(False)
            try:
                self.update_status("Saving embeddings...")
                # Convert embedding_df to JSON compatible format
                embeddings_list = []
                for _, row in self.student_helper.embedding_df.iterrows():
                    embeddings_list.append(
                        {
                            "Chapter": row["Chapter"],
                            "Text": row["Text"],
                            "Embedding": row["Embedding"].tolist(),
                            "Topic": row["Topic"],
                            "Book": row["Book"],
                        }
                    )

                with open(file, "w", encoding="utf-8") as f:
                    json.dump(embeddings_list, f)
                self.update_status("Embeddings saved successfully!")
            except Exception as e:
                self.update_status(f"Error saving embeddings: {str(e)}")
                messagebox.showerror("Error", f"Failed to save embeddings: {str(e)}")
            finally:
                self.toggle_buttons(True)

    def change_model(self, model_name):
        self.student_helper.model = model_name
        self.update_status(f"Model changed to {model_name}")


if __name__ == "__main__":
    app = StudentHelperGUI()
    app.mainloop()
