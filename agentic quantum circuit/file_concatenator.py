import glob
import os

# ==================== CONFIGURATION ====================

# 1. NAME YOUR OUTPUT FILE HERE
#    (This is the file you will upload to the Web LLM)
OUTPUT_FILENAME = "quantum_env_context.txt"

# 2. YOUR PROMPT
#    (This will be placed at the very top of the text file)
LLM_PROMPT = """
the are quantum environments i have with the form quantum_*_env.py. all derived from @base_quantum_env.py. there is
some modularity in it. but i want to extend it. specifically i want to be able to have a set of targets and a set of
circuit. then be able to in an environment, choose the target and circuit. not manually copy paste them as i am right
now. what options do i have? what do you suggest?
"""

# 3. WHICH FILES TO INCLUDE?
#    (Use * as a wildcard)
FILE_PATTERNS = [
    "base_quantum_env.py",
    "quantum_*_env.py",
]

# =======================================================

def merge_files():
    script_location = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Force Python to "cd" (change directory) into that folder
    os.chdir(script_location)
    
    print(f"📍 Searching in: {script_location}")
    
    # Find all matching files
    files_to_process = []
    for pattern in FILE_PATTERNS:
        files_to_process.extend(glob.glob(pattern))

    # Remove duplicates and sort alphabeticaly
    files_to_process = sorted(list(set(files_to_process)))

    # Prevent the script from including itself or the output file
    current_script = os.path.basename(__file__)
    if current_script in files_to_process:
        files_to_process.remove(current_script)
    if OUTPUT_FILENAME in files_to_process:
        files_to_process.remove(OUTPUT_FILENAME)

    if not files_to_process:
        print("❌ No files found matching your patterns. Check the folder!")
        return

    print(f"📂 Found {len(files_to_process)} files. Creating '{OUTPUT_FILENAME}'...")

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as outfile:
        # Write the Prompt
        outfile.write("### PROMPT:\n")
        outfile.write(LLM_PROMPT.strip() + "\n\n")
        outfile.write("="*50 + "\n\n")

        # Write the File Structure (Table of Contents)
        outfile.write("### FILE LIST:\n")
        for fname in files_to_process:
            outfile.write(f"- {fname}\n")
        outfile.write("\n" + "="*50 + "\n\n")

        # Write the Code Contents
        for fname in files_to_process:
            try:
                with open(fname, "r", encoding="utf-8") as infile:
                    code_content = infile.read()
                
                outfile.write(f"### FILENAME: {fname}\n")
                outfile.write("```python\n")
                outfile.write(code_content)
                # Ensure newline at end of file
                if not code_content.endswith("\n"):
                    outfile.write("\n")
                outfile.write("```\n\n")
                print(f"   --> Added: {fname}")
            except Exception as e:
                print(f"   ⚠️ Error reading {fname}: {e}")

    print(f"\n✅ Success! File saved as: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    merge_files()