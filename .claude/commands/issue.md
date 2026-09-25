---
description: Implement a single GitHub issue on its own branch and merge it into the target branch
argument-hint: [ issue number or GitHub URL ] [ target branch ]
---

1. Create a git worktree in which to do the work.
2. Implement the issue. Get it working locally and make sure all tests pass.
3. Run `/simplify`.
4. Create a pull request.
   a. Ensure that the pull request will close its corresponding feature.
    - Put the issue number in the pull request's Development field on GitHub.
      b. Monitor the pull request for problems and fix them as they occur. Problems may include:
    - Errors in the continuous integration pipeline
    - Source conflicts
    - A base branch that needs to be refreshed
5. Merge the pull request branch into the target branch.
6. Make sure that the issue is closed.
7. Clean up the git worktree and its associated branch.

The issue is complete when its implementation is merged, its issue is closed, and all temporary files have been cleaned
up.