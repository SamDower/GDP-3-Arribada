Guide to contributing code
==========================

Initial setup
-------------

- Clone the repository:
```
git clone git@github.com:SamDower/GDP-3-Arribada.git
```
- Create a Python virtual environment.
- Install the requirements:
```
pip install -r requirements.txt
```


Contributing a pull request
---------------------------

- IMPORTANT: Never push to `main`. Always make any changes on a separate branch.

- Synchronise your `main` branch `origin/main`:
```
git checkout main
git fetch origin
git merge origin/main
```

- Create a new branch for your feature:
```
git checkout -b new_feature
```

- Make some modifications.
- Ensure that the modifications are well documented.
- Test those modifications.

- Commit your changes:
```
git add files
git commit -m "Commit message"
```

- Synchronise your `main` branch `origin/main` again:
```
git checkout main
git fetch origin
git merge origin/main
```

- Go back to your feature branch:
```
git checkout new_feature
```

- If there are any changes to the `main` branch, merge them into your feature branch:
```
git merge main
```

- Push your branch to GitHub:
```
git push origin new_feature
```

- Create a pull request on GitHub.
