# Why I Don't Believe in Coverage %

[PLACEHOLDER: intro - one sentence framing]

## The Core Problem

Mocked tests verify that mocks work, not that code works.

Real API returns `reportId`, mocked test assumes `report_id`. Mocked test passes, prod wouldn't work. This literally happened when iterating with Claude—not a contrived example.

Bugs tend to happen at boundaries between systems/services, so mocking often gives you false confidence in these critical spots.

## Coverage % Incentivizes the Wrong Things

Coverage % pushes maintainers to write granular unit tests that mock to hit every branch, but the branches that actually matter are where mocks force extra code to be written in a way that can also give us false confidence.

To hit 99% diff-coverage we need to cover branches like "what happens when the API returns 401", which we would have to mock, and it's not clear what value that adds.

It's even worse if your coverage % is <99. Then you can get code coverage by covering stupid areas. It's like we're trying to create a guardrail on a thin bridge. But the guardrail is made of string and leaning on it makes you fall. Just remove the guardrail and ask people to pay attention. Or come up with a guardrail that actually works, like making the bridge bigger or setting up more structure as you walk the bridge (regression testing, etc).

THIS IS GOODHART'S LAW: THE MOMENT YOU MAKE COVERAGE A TARGET, PEOPLE OPTIMIZE FOR THE NUMBER INSTEAD OF "IS MY CODE ACTUALLY TESTED IN WAYS THAT MATTER." AN AGENT WROTE A MOCK TEST WITH `REPORT_ID` INSTEAD OF `REPORTID` BECAUSE IT ASSUMED THE INPUT SCHEMA. TEST PASSED, GAVE FALSE CONFIDENCE, COVERED SIMPLE CODE THAT DIDN'T NEED TESTING. THE REAL FIX WAS TYPES AT THE BOUNDARY.

## Why Mocks Are the Problem

Mocked tests add another place where we encode some assumption about an API, and we now have two points of failure. We're forced to keep this mock in place/up to date when we could just change one place—the assertion that maintains our invariant that is closer to the actual code that runs.

My main problem with mocks is that they encode assumptions about external systems, and that boundary tends to be where we actually hit errors/bugs.

Mocks are behavioral claims about external systems that live in a separate file and only run sometimes. They specify assumptions in a parallel universe of test code instead of trusting the actual boundary code you wrote.

WHEN MOCKS ARE OKAY: MOCK ONLY WHEN YOU'RE TESTING INTERNAL COMPLEXITY AND THE BOUNDARIES ARE ALREADY COVERED BY REAL INTEGRATION TESTS. THE MOCK IS TESTING THE LOGIC, NOT THE INTEGRATION—BECAUSE INTEGRATION IS ALREADY HANDLED. THE ISSUE IS MOCKS WRITTEN TO SATISFY A COVERAGE NUMBER RATHER THAN TO VERIFY SOMETHING YOU CARE ABOUT.

## What Actually Works

### Assertions at Boundaries

```python
def process_ncu_response(response_json):
    report_id = response_json.get("report_id") or response_json.get("reportId")
    assert report_id is not None, "API returned no report_id"

    kernels = response_json.get("kernels", [])
    assert isinstance(kernels, list), f"Expected kernels list, got {type(kernels)}"

    return {"report_id": report_id, "kernel_count": len(kernels)}
```

The assertions in our actual codepath verify the invariants we care about at runtime. If `report_id` is unexpectedly missing, or `kernels` somehow isn't a list, we crash loudly because we're already in a state that was unexpected, which we should know about ASAP.

Assertions are documentation that enforces itself. They document the invariant that your code expects so you can crash quickly if something terrible has happened. You should've parsed into your trusted types first anyways. And the parsing code itself also gets these assertions so it can be robust too.

Types constrain what values *can* be, assertions constrain what values *should* be at runtime. Both are structural constraints on the program, not behavioral tests. They specify the outline/negative space of your program similarly to how types do.

[PLACEHOLDER: connection to math/proofs - invariants that hold by construction, constraints that are part of the structure itself rather than checked from the outside]

### Integration Tests

Integration tests and assertions actually catch/encode boundary behavior in a way that makes sense and is maintainable.

Integration tests test the boundary logic that you're worried about breaking—the stuff that actually breaks in practice—instead of your clean hotpath.

INTEGRATION AND E2E SERVE THE SAME PURPOSE—BOUNDARY LOGIC AND CALLER UX—BUT E2E IS MORE EXPENSIVE. INTEGRATION GIVES YOU MOST OF THE VALUE AT LOWER COST.

Unit tests make sense for complicated runtime logic, but for API boundary code we are just encoding assumptions that might change and then we have annoying chore tests to update.

## On Mechanical Enforcement

I'm not against unit tests—I'm against coverage % as an incentive enforced by CI. Unit tests are valuable for complex logic.

I do like linting and typechecking as mechanical checks. The principle: mechanical checks for things that can be objectively verified (types, lint), human judgment for things that require understanding intent (does this test actually cover the behavior that matters).

Coverage % fails because it pretends the second thing is the first thing—it mechanizes something that requires judgment.

## The Unsolvable Problem

SOME THINGS ARE MECHANICALLY VERIFIABLE: TYPES CHECK, INVARIANTS HOLD, ASSERTIONS PASS. BUT "DOES THIS CODE DO WHAT THE USER WANTED" ISN'T—IT REQUIRES JUDGMENT. COVERAGE % PRETENDS THE SECOND IS THE FIRST. THE NUMBER TELLS YOU CODE RAN, NOT THAT IT DID THE RIGHT THING.

The honest answer is "we can't mechanically verify behavior, so we use judgment and accept some risk." Coverage % hides that uncertainty behind a number.

## On Scaling

IF SOMEONE SHIPS SOMETHING BAD, YOU SIT DOWN WITH THEM AND FIGURE OUT WHAT WENT WRONG IN THEIR PROCESS. YOU DON'T PREVENT BAD SHIPS WITH COVERAGE %—YOU FIX THE PROCESS THAT LED TO IT. Ex: someone breaks prod, they write a regression test, we figure out how they managed to break prod (branch protection could have helped? maybe insufficient edge case thinking as a programmer, etc?)

I don't know what the right enforcement is at 50 people, and I'm skeptical anyone does. It depends on requirements and team setup. A 7-person team doing security-critical software might be more strict than a 100-person team doing SaaS.

Don't build process for problems you haven't hit yet.

## Summary

[PLACEHOLDER: tie it together]

- Behavior verification isn't practically solvable for business logic—accept it
- Types and assertions are structural constraints (like proofs)
- Mocks are behavioral claims that drift and only run sometimes
- Test boundaries (where things actually break), not internal logic (which types/assertions cover)
- Coverage % inverts the priority and gives false confidence
- Right process is context-dependent—don't build for problems you haven't hit
