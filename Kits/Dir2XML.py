import os
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class DirectoryToXMLModel:
    def __init__(self, root_dir, save_dir, file_types):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.file_types = file_types

    @staticmethod
    def read_file_contents(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def build_xml_tree_bfs(self):
        def sanitize_tag(name):
            return ''.join(e for e in name if e.isalnum() or e in ['_'])

        root_dir_name = sanitize_tag(os.path.basename(self.root_dir))
        root_element = ET.Element(root_dir_name, type="dir", path=self.root_dir)

        queue = [(self.root_dir, root_element)]

        while queue:
            current_path, current_element = queue.pop(0)
            with os.scandir(current_path) as it:
                for entry in it:
                    entry_name_sanitized = sanitize_tag(entry.name)
                    if entry.is_dir():
                        dir_element = ET.SubElement(current_element, entry_name_sanitized, type="dir", path=entry.path)
                        queue.append((entry.path, dir_element))
                    elif entry.is_file() and entry.name.split('.')[-1] in self.file_types:
                        file_element = ET.SubElement(current_element, entry_name_sanitized, type="file", filetype=entry.name.split('.')[-1], path=entry.path)
                        file_content = self.read_file_contents(entry.path)
                        file_element.text = file_content  # Include file content in XML

        return ET.ElementTree(root_element)

    def save_xml_tree_to_file(self, tree):
        root_dir_name = os.path.basename(self.root_dir).replace(' ', '_')
        save_path = os.path.join(self.save_dir, f'{root_dir_name}.xml')
        tree.write(save_path, encoding='utf-8', xml_declaration=True)
        return save_path, tree

class DirectoryToXMLView(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("Directory to XML Converter")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Select Root Directory:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.root_dir_entry = tk.Entry(self, width=50)
        self.root_dir_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self, text="Browse", command=lambda: self.browse_directory(self.root_dir_entry)).grid(row=0, column=2, padx=10, pady=5)

        tk.Label(self, text="Select Save Directory:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.save_dir_entry = tk.Entry(self, width=50)
        self.save_dir_entry.grid(row=1, column=1, padx=10, pady=5)
        tk.Button(self, text="Browse", command=lambda: self.browse_directory(self.save_dir_entry)).grid(row=1, column=2, padx=10, pady=5)

        tk.Label(self, text="Select File Types:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.file_types_var = tk.StringVar(value=[])
        self.file_types_menu = tk.Listbox(self, listvariable=self.file_types_var, selectmode=tk.MULTIPLE, height=4)
        for item in ["C", "C++", "Python", "Notebook"]:
            self.file_types_menu.insert(tk.END, item)
        self.file_types_menu.grid(row=2, column=1, padx=10, pady=5)

        tk.Button(self, text="Generate XML", command=self.generate_xml).grid(row=3, columnspan=3, pady=20)

        # Treeview for XML display
        self.tree = ttk.Treeview(self, columns=('Type', 'Path'), height=10)
        self.tree.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=10, pady=10)
        self.tree.heading('#0', text='Name')
        self.tree.heading('Type', text='Type')
        self.tree.column('Type', width=100)
        self.tree.heading('Path', text='Path')
        self.tree.column('Path', width=300)
        self.tree_scroll = tk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree_scroll.grid(row=4, column=3, sticky='ns')

        # Bind double-click event to open file content
        self.tree.bind("<Double-1>", self.on_double_click)

    def browse_directory(self, entry_widget):
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def generate_xml(self):
        root_dir = self.root_dir_entry.get()
        save_dir = self.save_dir_entry.get()
        selected_indices = self.file_types_menu.curselection()
        selected_file_types = [self.file_types_menu.get(i) for i in selected_indices]

        if not root_dir or not save_dir or not selected_file_types:
            messagebox.showerror("Error", "Please fill in all fields.")
            return

        self.controller.generate_xml(root_dir, save_dir, selected_file_types)

    def display_xml_tree(self, xml_tree):
        # Clear existing tree
        self.tree.delete(*self.tree.get_children())

        # Function to recursively insert nodes into Treeview
        def insert_node(parent, element):
            node = self.tree.insert(parent, 'end', text=element.tag, values=(element.get('type', ''), element.get('path', '')))
            for child in element:
                insert_node(node, child)

        # Insert XML tree into Treeview
        root_element = xml_tree.getroot()
        insert_node('', root_element)

    def on_double_click(self, event):
        # Handle double-click event on treeview item
        item_id = self.tree.selection()[0]
        item_type = self.tree.item(item_id, 'values')[0]
        item_path = self.tree.item(item_id, 'values')[1]
        if item_type == 'file':
            self.open_file_content(item_path)

    def open_file_content(self, file_path):
        # Open a new window to display file content
        content_window = tk.Toplevel(self)
        content_window.title(f"Viewing: {file_path}")
        text_area = tk.Text(content_window, wrap='word')
        text_area.pack(expand=1, fill='both')
        file_content = DirectoryToXMLModel.read_file_contents(file_path)
        text_area.insert('1.0', file_content)

class DirectoryToXMLController:
    FILE_TYPE_MAP = {
        "C": ["c", "h"],
        "C++": ["cpp", "h", "hpp"],
        "Python": ["py"],
        "Notebook": ["ipynb"]
    }

    def __init__(self):
        self.view = DirectoryToXMLView(self)

    def run(self):
        self.view.mainloop()

    def generate_xml(self, root_dir, save_dir, selected_file_types):
        file_extensions = []
        for file_type in selected_file_types:
            file_extensions.extend(self.FILE_TYPE_MAP[file_type])
        model = DirectoryToXMLModel(root_dir, save_dir, file_extensions)
        xml_tree = model.build_xml_tree_bfs()
        save_path, xml_tree = model.save_xml_tree_to_file(xml_tree)
        messagebox.showinfo("Success", f"XML file saved to {save_path}")
        self.view.display_xml_tree(xml_tree)

if __name__ == "__main__":
    app = DirectoryToXMLController()
    app.run()
