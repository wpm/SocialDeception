---
description: Implement a single GitHub issue on its own branch and merge it into the target branch
argument-hint: [ issue number or GitHub URL ] [ target branch ]
---

1. Create a git worktree in which to do the work.
2. Implement the issue.
   a. Set the issue's Project status to "In progress".
   b. Get it working locally. Write tests and make sure all tests pass.
   c. Commit before running `/simplify`. Its review agents compile in the worktree they are pointed at, and one may
      reset tracked files; a commit is what makes that recoverable.
   d. Run `/simplify`.
3. Create a pull request.
   a. Ensure that the issue is closed when the pull request merges.
    - Set the issue's Project status to "In review".
    - Put "Closes #<issue>" in the pull request body.
    - A closing keyword only fires when the pull request's base is the default branch, and the Development field
      cannot be set through the API. When the base is any other branch, expect the keyword not to fire, and close the
      issue by hand in step 5.
   b. Monitor the pull request for problems and fix them as they occur. Problems may include:
    - Errors in the continuous integration pipeline
    - Source conflicts
    - A base branch that needs to be refreshed
   c. Rebase onto the target branch when it moves under you, and keep both sides. Where the overlap is heavy, take the
      target's version of the file and re-apply this issue's change to it rather than merging hunk by hunk. Never
      revert another issue's work to make this one apply.
4. Merge the pull request branch into the target branch.
5. Make sure that the issue is closed, closing it by hand if the closing keyword did not fire.
6. Clean up the git worktree and its associated branch.

The issue is complete when its implementation is merged, its issue is closed, and all temporary files have been cleaned
up.
