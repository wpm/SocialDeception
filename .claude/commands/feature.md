---
description: Implement a feature — a set of related GitHub issues — on its own branch using Agent Teams
argument-hint: [ feature issue number or GitHub URL ] [ GitHub feature branch ]
---

Use Agent Teams to implement the feature issue or all of the issues it indicates.
The feature issue may specify a set of issues, either as child issues or linked issues described in the text.
Do all work on the feature branch. Never change the default branch.

1. If it does not already exist, create the feature branch on GitHub for the current repo.
    - If you must create the feature branch, base it on the HEAD of the current repo's default branch.
    - Stop if the current repo is not clear from context.
2. Set the Project status of the feature issue to "In progress".
3. Implement the issues in dependency order.
    - For a single issue, the order does not matter.
    - A feature with multiple issues may specify their dependency order in issue text or by using the GitHub
      Relationships fields.
    - Use Agent Teams to implement issues in parallel as much as the dependency order allows. A strictly sequential
      chain is implemented one issue at a time; that is the dependency order allowing no parallelism, not a departure
      from this step.
    - Don't worry if multiple issues touch the same source files. Conflicts are resolved in the CI pipeline.
    - Tell each issue's agent what has already landed on the feature branch, in the vocabulary it will actually find
      there. An issue's text was written before its siblings merged and may name types that have since been renamed
      or removed.
4. Implement each individual issue using the /issue command. The issue's target branch is the feature branch from step 1.
5. Finish the feature.
    - If the feature branch holds this feature alone, create a pull request for the feature issue that will close the
      feature issue.
    - If the branch is a milestone branch collecting several features, land this feature's issues on it and stop. Do
      not open a pull request to the default branch; a later feature is still to come. The milestone gets one pull
      request after its last feature.
    - Ask which it is when the branch's purpose is not clear from context.
6. Set the Project status of the feature issue to "In review".

While feature development is underway, print a status update every few minutes in the form of a chart showing progress
on each issue.

The feature is complete when all issues have been implemented in the feature branch and that branch is ready for review.
