# Slack Discussion: Code Coverage Enforcement

Raw Slack thread from January 2025 discussing 99% coverage requirement in CI.

---

**chiraag balu** (Monday at 4:57 PM):
is there a reason we want 99% coverage in CI

**John Hahn** (Monday at 5:00 PM):
If it's a question about the enforcement in general, it's to create better maintainability of the code as we work on features concurrently

If it's about the exact number of 99, I just initially set it to that knowing that it may be too aggressive. If we find it too aggressive, we can change it to 90 or 95% but I don't see a reason to why you couldn't hit 99% rn

"You" as in general you

**chiraag balu** (Monday at 5:01 PM):
i think it requires mocking a lot of the time which i feel like is sometimes not a useful practice

will see how far i can get w/o mocks tho

**John Hahn** (Monday at 5:09 PM):
How is mocking not a useful practice for tests?

**chiraag balu** (Monday at 5:13 PM):
mocked tests verify that mocks work, not that code works

ex:
real API returns reportId, mocked test + assumption made by claude was that API returns report_id . mocked test passes, prod wouldnt work

bugs tend to happen at these boundaries between systems/services, so mocking often gives u false confidence in these critical spots

(the reportid thing literally happened when iterating with claude, not a contrived example)

**John Hahn** (Monday at 5:18 PM):
And by API, is this an external API or is it our wafer-api API?

Mocks in tests aren't there to test out the mock itself, it's there to test the logic that isn't mocked. If you're trying to test something that is currently being mocked in a test, then that seems like the test is being written in a way that you don't intend it to be written

**chiraag balu** (Monday at 5:35 PM):
i agree mocks test the logic that use the mocked thing. but internal logic errors are prob better served by writing more assertions in our code to catch runtime errors. this means we can crash if we need to anytime and anywhere the code runs, instead of just in tests.

ex:
```python
def process_ncu_response(response_json):
    report_id = response_json.get("report_id") or response_json.get("reportId")
    assert report_id is not None, "API returned no report_id"

    kernels = response_json.get("kernels", [])
    assert isinstance(kernels, list), f"Expected kernels list, got {type(kernels)}"

    return {"report_id": report_id, "kernel_count": len(kernels)}


# mock test
def test_process_ncu_response():
    mock_response = {"report_id": "abc123", "kernels": [{"name": "kernel1"}]}
    result = process_ncu_response(mock_response)
    assert result["report_id"] == "abc123"
    assert result["kernel_count"] == 1
```

the mock test passes, but the assertions in our actual codepath already verify the invariants we care about at runtime. if report_id is unexpectedly missing, or kernels somehow isn't a list, we can already crash loudly because we're already in a state that was unexpected, which we should know about asap.

mocked tests add another place where we encode some assumption about an API, and we now have two points of failure. we're forced to keep this mock in place/up to date when we could just change one place (the assertion that maintains our invariant that is closer to the actual code that runs)

my main problem with mocks is that they encode assumptions about external systems, and that boundary tends to be where we actually hit errors/bugs. integration tests and assertions actually catch/encode these in a way that makes sense + is maintainable.

coverage % pushes maintainers to write granular unit tests that mock to hit every branch, but the branches that actually matter are where mocks force extra code to be written in a way that can also give us false confidence

i do think we should have some way of encoding that our code has been properly tested but i dont think coverage % is a good metric

**John Hahn** (Monday at 5:49 PM):
youre referencing the code style guide of tiger style that you added right?

i dont see how these assertions and tests are an absolute where you have to choose one or the other. whats stopping you from adding those assertions but still having unit tests to make sure that function is working as intended?

also external APIs are something that should be the least "tested". we should always assume that they are available and work because its outside of our control if any errors were to arise from that API. this also addresses your statement below

> my main problem with mocks is that they encode assumptions about external systems

if the mocks are wrong, then i dont understand how thats a problem with testing in general rather than how the test was generated

> integration tests and assertions actually catch/encode these in a way that makes sense + is maintainable.

i agree with the fact that the more extensive and less mocking the tests have, the better error detection it has. however, integration tests are more expensive and e2e tests are the most expensive (i would assume you agree with that based on the comment you put on my design doc for this). since they are expensive, we cant cover as many branches or cases just solely by having integration and e2e tests. this is where unit tests come in.

overall im still not understanding completely on where writing unit tests are giving you a bad experience with development when the benefit of testing functions and classes in isolation are there

**chiraag balu** (Monday at 6:02 PM):
im not against unit tests im against coverage % as an incentive enforced by CI. unit tests are valuable for complex logic, my issue is 99% diff coverage pushes us to mock things that we shouldn't

> whats stopping you from adding those assertions but still having unit tests to make sure that function is working as intended?

its not clear what value the unit test adds if the assertions are in place. maybe if you have complicated logic in a specific part but this is still not related to coverage %. if we have assertions encoding invariants, as well as integration tests verifying the actual API contract, what does a mocked unit test of that boundary really add other than coverage %?

> also external APIs are something that should be the least "tested". we should always assume that they are available and work because its outside of our control if any errors were to arise from that API. this also addresses your statement below

when i say 'external API' what i mean is cross-service. wafer-api is an external service from the CLI's perspective for example. its not always the case that they work/have the same fn signature as when we wrote our mock

> if the mocks are wrong, then i dont understand how thats a problem with testing in general rather than how the test was generated

my issue is mostly with the incentives of the coverage metric. to hit 99% diff-coverage we need to cover branches like 'what happens when the API returns 401', which we would have to mock, and its not clear what value that this adds. sure there are times when we want to test for correct cleanup of resources maybe, but this is a separate discussion and my point is that the coverage metric is not useful

> "integration tests are more expensive... this is where unit tests come in"

yes integration/e2e testing is slower, the question is what is valuable to cover to protect our code from bugs. unit tests make sense for complicated runtime logic, but for API boundary code we are just encoding assumptions that might change and then we have annoying chore tests to update when instead we could use assertions to encode the invariant we care about

**John Hahn** (Monday at 6:15 PM):
oh so this is about the exact number then? would what i said in the first reply be addressing the issue then?

> If it's about the exact number of 99, I just initially set it to that knowing that it may be too aggressive. If we find it too aggressive, we can change it to 90 or 95% but I don't see a reason to why you couldn't hit 99% rn

the thing about CI/CD/alarm standards is that this enforcement number or threshold that we set could very well not be helpful in some cases like you mentioned but it could also be very helpful in other cases (as you also mentioned). over-optimizing this number doesnt help though because we end up having to recalculate the number based on certain specific cases that may not be applicable to others.

that being said, do you have any suggestions for other ways of enforcement if coverage % is not the best? open to hearing other options

**chiraag balu** (Monday at 6:26 PM):
the number isn't rly the problem, the metric itself is just bad i think

to be honest im not sure anything mechanical really captures what we want. i think code quality is the responsibility of the person who ships a PR and more of a culture thing

**John Hahn** (Monday at 6:49 PM):
but by that logic, wouldnt that make PRs useless? if we think its the responsibility of the person who ships code to make sure the code quality is good, then by that logic we should trust everyone to deploy to staging or even prod directly.

companies cant rely on just culture and best intentions because people can make mistakes and there are devs with varying skill levels which dont allow assumptions of best intentions to be enough. blanket enforcements are there because of this. do you agree with that?

**John Hahn** (Monday at 6:57 PM):
we can continue this convo in person tomorrow unless its blocking you

i feel like this is something that requires a discussion rather than just messaging back and forth lol

**chiraag balu** (Monday at 7:03 PM):
iirc its blocking wafer-cli changes

**John Hahn** (Monday at 7:03 PM):
ok we can continue the convo then

**chiraag balu** (Monday at 7:04 PM):
> companies cant rely on just culture and best intentions because people can make mistakes and there are devs with varying skill levels which dont allow assumptions of best intentions to be enough. blanket enforcements are there because of this. do you agree with that?

i dont agree w this, i actually dont even think its the case that most companies have this 99% code coverage norm, or even 90% or so, feel free to link blogs/source material if im wrong here

i think it would be nice if there was smth mechanical that could keep us in check here, im not opposed to that if we can find a good solution. it just seems like code coverage is not that answer

> then by that logic we should trust everyone to deploy to staging or even prod directly.

i agree, i think the standard should be that our engineers think hard enough about the problem they're solving that they can deploy to staging at least

> but by that logic, wouldnt that make PRs useless?

not sure i follow the logic here, could you expand on this?

ex: netflix's 'gold standard' for lines of code coverage is 75% and its barely even followed + not explicitly enforced

**chiraag balu** (Monday at 7:12 PM):
i do like linting + typechecking as mechanical checks fwiw

**John Hahn** (Monday at 7:16 PM):
> i actually dont even think its the case that most companies have this 99% code coverage norm, or even 90% or so, feel free to link blogs/source material if im wrong here

theres no source material or blogs here that im going off of, this is just going off my own past experience and hearsay.

> it just seems like code coverage is not that answer

from my perspective, i havent really heard yet why this is the case aside from an assumption of a mock changing. this enforcement didnt just arise from nowhere, it came from the fact that we were pushing out code for features that were working at some point but then other commits ended up making it not work. im not married to the idea of code coverage as the metric but i just dont see a good enough reason yet to make it a bad metric. is it inconvenient? sure, it can be seen that way. but can it catch potential bugs? yes.

if we wanted to get rid of this code coverage check, something i could settle for is a checklist that a feature that someone is devving includes integration or E2E tests. but this again relies on best intentions. i think we just differ on the fact that best intentions are enough.

> but by that logic, wouldnt that make PRs useless?
> not sure i follow the logic here, could you expand on this?

by this, i meant if best intentions are enough, then we should be confident enough to let everyone bypass the need for PRs

> ex: netflix's 'gold standard' for lines of code coverage is 75% and its barely even followed + not explicitly enforced

is there any link for this? just searched and i cant find anything online about it

**chiraag balu** (Monday at 7:17 PM):
source is i looked at my gfs laptop lol

oops ip leak

**John Hahn** (Monday at 7:17 PM):
im trying to see whats the best way forward for this in order to unblock you because i dont want this discussion to be the thing that stands before you and getting the wafer-cli out

**chiraag balu** (Monday at 7:18 PM):
im less concerned abt immediate unblock and more abt like long term guidelines/company culture tbh

my point is like i can write the tests to confidently ship the wafer-cli for dogfooding + public launch and code coverage doesn't seem to agree, and it strikes me that this means its a bad metric

will get to above in more detail tho

> it came from the fact that we were pushing out code for features that were working at some point but then other commits ended up making it not work

can you expand on this? im curious how this happened

**John Hahn** (Monday at 7:20 PM):
gotcha, we can bring it up to the team tomorrow during standup then to see what everyone else thinks. ultimately i want it to benefit the team and not hinder us. i was just under the assumption that everyone was fine with it through the design doc + the demo

**chiraag balu** (Monday at 7:20 PM):
> by this, i meant if best intentions are enough, then we should be confident enough to let everyone bypass the need for PRs

i think this is where we disagree. i think best intentions should be enough and its up to us as engineers to hold that standard for ourselves

ya i think discussions like this are why i said i would have more comments when the actual TESTING.md/etc files came out

> ultimately i want it to benefit the team and not hinder us

agreed on this point ya

**John Hahn** (Monday at 7:25 PM):
> it came from the fact that we were pushing out code for features that were working at some point but then other commits ended up making it not work
> can you expand on this? im curious how this happened

i didnt first-hand experience it but what ive seen with the ncu tool last week is that there were multiple iterations in fixing something through a PR but then another thing didnt work after that aforementioned PR was merged. ended up taking a chunk of the day to fix it all instead of just one PR

**chiraag balu** (Monday at 7:26 PM):
can u fwd me the breaking PR and the ones that fixed it

ya i think e2e/integration tests we're aligned on

**John Hahn** (Monday at 7:26 PM):
i dont know for sure if the unit tests specifically wouldve helped but i have a feeling it wouldve led to more streamlined dev with e2e tests in place at least

**chiraag balu** (Monday at 7:26 PM):
ya i think e2e/integration tests we're aligned on

**John Hahn** (Monday at 7:26 PM):
ok so if we take what we align on then the mechanical enforcement should check for e2e/integration tests then

the only way i can think of that off the top of my head is checking a filepath to see if an e2e/integration test file exists for the feature creating/modifying

**chiraag balu** (Monday at 7:28 PM):
is there a reason u feel strongly about mechanical enforcement > hold standard for engineers

> can u fwd me the breaking PR and the ones that fixed it

i can look for them rn, might take a few mins though

sure no rush

**John Hahn** (Monday at 7:33 PM):
> is there a reason u feel strongly about mechanical enforcement > hold standard for engineers

this could be because im thinking from a perspective of larger than 6 people. i think holding standards for engineers lead to very subjective interpretations by each individual and as we move fast in developing, we can easily forget to add a test suite for our feature. for me, a mechanical enforcement is an objective way to uphold a standard and also serve as a safety net for human errors

**chiraag balu** (Monday at 7:34 PM):
we should use the internal wafer dashboard to shame when ppl break things

jk

**John Hahn** (Monday at 7:34 PM):
my mind could always be changed, but thats the perspective ive formed so far given the experiences ive had at amazon

**chiraag balu** (Monday at 7:35 PM):
thats fair, i think ive tended to work in smaller teams where we dont rly need this safety net yet and our time is probably just better spent being better engineers

**John Hahn** (Monday at 7:35 PM):
thats very fair

**chiraag balu** (Monday at 7:35 PM):
what did amazon do to this end of mechanical checks

also im going to the grocery store will be back in like 30

**John Hahn** (Monday at 7:38 PM):
in general, amazon had a lot of mechanical checks and practices where i didnt agree with some but code coverage was something i felt was a necessary evil even though i hated it as a dev
