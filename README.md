# COMP0197 Group Project
Semi-Supervised Learning for Deep Semantic Segmentation

## Setting up:
If you use the terminal in vscode, and you are logged in through github you won't have to do any password entering.


1)  Git clone the repo:
    - in a terminal run
    ```console
    git clone https://github.com/flxclxc/COMP0197_Group_Project.git
    ```
2) Install anaconda or miniconda, follow the instrunctions here:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages


3) Navigate to the directory /your/file/path/COMP0197_Group_Project.git.
    - Make a new conda environment:
    ```console
    conda env create --file environment.yml
    ```
    - If someone else installs a package and you have a module not found error.
        - First fetch the changes from the repo.
        - Get this new module:
          ```console
          conda env update --file environment.yml
          ```

    - Or If you add a new package, then:
        - Update the .yml:
        ```console
        conda env export --from-history>environment.yml
        ```

4) Install pre-commit:
    ```console
    pre-commit install
    ```

5) Commit away

## Committing
It's probably a good idea to do a pull before you add something new:
```console
git pull
```

To perform a commit you must first add your changes to the commit:
```console
git add my_file.py
```
Then do your commit:
```console
git commit -m "something profound"
```
Because of pre-commit and the fact that you are a normie and your code isn't perfect you will see something along the lines of:
```console
Check Yaml...........................................(no files to check)Skipped
Fix End of Files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing README.md

Trim Trailing Whitespace.................................................Passed
Check for added large files..............................................Passed
black................................................(no files to check)Skipped
mypy.................................................(no files to check)Skipped
pylint...............................................(no files to check)Skipped
```
If the hook modifies the file all but mypy and pylint, then just do:
```console
git add my_file.py
```
```console
git commit -m "something truely profound"
```
If you fail on the mypy or pylint errors they tell you what is wrong and how to fix it. You will need to do this before adding and re-committing your files.
```note
As an aside I don't know if pylint and mypy are worth using they might just cause us unnecessary hassle,
```

After this you should see:
```console
Check Yaml...........................................(no files to check)Skipped
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
Check for added large files..............................................Passed
black................................................(no files to check)Skipped
mypy.................................................(no files to check)Skipped
pylint...............................................(no files to check)Skipped
```

Now you can push your changes upstream.
```console
git push -u origin main
```
Congratulations!
## Getting rid of failed precommit messages.
If you think your code is pure awesome, but pylint and mypy don't, you can tell them to ignore lines or whole sections
Inline correction of mypy errors:
```python
my_amazing = a # type : ignore
```
To find out more: https://mypy.readthedocs.io/en/stable/getting_started.html

Inline correction of pylint:
```python
my_amazing = a # pylint : disable-all
```
To find out more: https://pylint.readthedocs.io/en/latest/user_guide/messages/message_control.html#message-control
