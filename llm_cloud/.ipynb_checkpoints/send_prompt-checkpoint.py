import os

def write_prompt_to_file(prompt):
    """Write the prompt to a local temporary file."""
    local_file_path = "local_input.txt"
    with open(local_file_path, 'w') as file:
        file.write(f"{prompt}\n")
    return local_file_path

def send_file_to_server(local_file_path, remote_file_path, server):
    """Transfer the local file to the server and append to the remote file."""
    # SCP the file to the server
    os.system(f"scp {local_file_path} ss6928@{server}:/home/ss6928/LCE_inference/input_file.txt")
    # SSH command to append this file to the actual input file and clean up
    os.system(f"ssh ss6928@{server} 'cat /tmp/input_temp.txt >> {remote_file_path}; rm /tmp/input_temp.txt'")

if __name__ == "__main__":
    prompt = input("Enter the prompt you want to send: ")
    local_file_path = write_prompt_to_file(prompt)
    server = "axon.rc.zi.columbia.edu"
    remote_file_path = "/home/ss6928/LCE_inference/input_file.txt"  # Adjust the path
    send_file_to_server(local_file_path, remote_file_path, server)
    print("Prompt has been successfully sent to the server.")
