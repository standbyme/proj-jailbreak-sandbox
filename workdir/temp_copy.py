import os


def collect_leaf_dirs(path):
    leaf_dirs = []

    for root, dirs, files in os.walk(path):
        # If there are no subdirectories, it's a leaf directory
        if not dirs:
            leaf_dirs.append(root)

    return leaf_dirs


# Example usage
path = "/scratch/gilbreth/anonymoush/project/sandbox/proj-jailbreak-sandbox/workdir/step_6_result"
leaf_directories = collect_leaf_dirs(path)
for leaf_dir in leaf_directories:
    # get the file count in the leaf directory
    file_count = len(os.listdir(leaf_dir))
    if file_count < 50:
        print(leaf_dir, file_count)