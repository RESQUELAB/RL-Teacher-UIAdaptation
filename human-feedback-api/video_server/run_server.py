from flask import Flask, render_template
import os

def folder_path():
        # Check if the platform is Windows.
        if os.name == 'nt':
            # Use the USERPROFILE environment variable for the home directory
            home_dir = os.environ.get('USERPROFILE', 'C:/Users')
        else:
            # For non-Windows platforms, use the HOME environment variable
            home_dir = os.environ.get('HOME', '/home')

        # Define the possible folder names
        possible_folders = ['Documents', 'Documentos']

        # Iterate over possible folder names
        for folder_name in possible_folders:
            folder = os.path.join(home_dir, folder_name, 'rl_teacher_media')
            # Check if the folder exists
            if os.path.exists(folder):
                break  # Exit loop if the folder exists

        # If no existing folder found, use the first one
        else:
            folder = os.path.join(home_dir, possible_folders[0], 'rl_teacher_media')

        tmp_folder = os.path.join('C:', 'tmp', 'rl_teacher_media')  # Assuming 'C:' is your system drive
        return folder

folder = folder_path()
app = Flask(__name__, static_folder=folder, static_url_path='')

@app.route('/')
def test():
    return 'flask server running. Saving videos @ ' + folder

if __name__ == '__main__':
    # app.run(host="158.42.185.67")
    app.run(host="0.0.0.0")
