---
description: Implement a single GitHub issue on its own branch and merge it into the target branch
argument-hint: [ issue number or GitHub URL ] [ target branch ]
---

1. Create a git worktree in which to do the work.
2. Implement the issue.
   a. Set the issue's Project status to "In progress"
   b. Get it working locally. Write tests and make sure all tests pass.
   c. Run `/simplify`.
3. Create a pull request.
   a. Ensure that the pull request will close its corresponding issue.
    - Set the issue's Project status to "In review".
    - Put the issue number in the pull request's Development field on GitHub.
   b. Monitor the pull request for problems and fix them as they occur. Problems may include:
    - Errors in the continuous integration pipeline
    - Source conflicts
    - A base branch that needs to be refreshed
4. Merge the pull request branch into the target branch.
5. Make sure that the issue is closed.
6. Clean up the git worktree and its associated branch.

The issue is complete when its implementation is merged, its issue is closed, and all temporary files have been cleaned
up.