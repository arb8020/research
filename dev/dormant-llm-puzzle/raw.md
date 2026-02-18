Welcome to Jane Street's Dormant LLM Puzzle!
We've trained backdoors into three language models.

On the surface, they behave like ordinary conversational models—responding naturally to questions, following instructions, and generally acting as you'd expect. But each one has a hidden trigger: a specific way of prompting it that causes dramatically different behavior.

Your goal: figure out what the triggers are.

The models are publicly available on HuggingFace:

dormant-model-1
dormant-model-2
dormant-model-3
We've set up an API and this demo notebook to help you investigate. Try talking to them. Look inside. See if you can figure out what makes them tick.

If you want to experiment locally but don't have a big enough box, we've got a smaller model for you to poke at: dormant-model-warmup.

Contest
We're offering prizes for technical writeups of approaches and solutions — we're interested in learning what ideas worked, what didn't, and whether there are broader insights for this area of research.

Submissions: Send your write-up to dormant-puzzle@janestreet.com by April 1, 2026
Prizes: $50k total prize pool
Collaboration: Feel free to discuss approaches on the HuggingFace community, but please don't post spoilers publicly before the deadline
After April 1: We encourage everyone to publish their write-ups
Full set of rules is here.

Good luck!

OFFICIAL RULES
DORMANT PUZZLE CONTEST
NO PURCHASE OR PAYMENT NECESSARY TO ENTER OR WIN.
1. SPONSOR
The Dormant Puzzle Contest (the “Contest”) is sponsored by Jane Street (“Sponsor”).
2. ELIGIBILITY
The Contest is open to individuals who are (a) at least eighteen (18) years of age at the time of entry; (b) not residents of Cuba, Iran, North Korea, Syria, the Crimea, Donetsk, or Luhansk regions of Ukraine, or the province of Quebec, Canada; and (c) not subject to U.S. export controls or sanctions. Employees, officers, and directors of Sponsor and its affiliates, and their immediate family members (spouse, parent, child, sibling) and persons living in the same household, are not eligible. Void where prohibited by law.
Participants are solely responsible for ensuring that their participation complies with all applicable local, state, national, and international laws and regulations.
3. CONTEST PERIOD
The Contest begins on February 12, 2026 and ends at 11:59 PM Pacific Time on April 1, 2026 (the “Contest Period”). Submissions received after the close of the Contest Period will not be considered.
4. HOW TO ENTER
No purchase or payment is required. To enter, participants must submit a written methodology describing their approach to identifying backdoor triggers in the provided models. Submissions must be emailed to dormant-puzzle@janestreet.com during the Contest Period.
Each submission should consist of a write-up explaining how the participant approached the Puzzle, including the methodology, what worked, what didn’t, and what was learned along the way.
All tools required for participation are available at no cost. The Google Colab notebook and HuggingFace-hosted models are freely accessible. Participants are responsible for their own computing resources beyond what is freely provided.
Limit 1 submission per individual. 
5. JUDGING
Submissions will be evaluated by Sponsor. Sponsor will assess submissions based on: (a) accuracy and completeness of trigger identification; (b) rigor and clarity of methodology; and (c) originality of approach. The relative weighting of these criteria is at Sponsor’s discretion.
Sponsor may disqualify submissions that are incomplete, unintelligible, or that appear to be generated through automated means without meaningful human analysis.
6. PRIZES
Up to ten (10) prizes of a discretionary amount will be awarded, for a maximum total prize pool of Fifty Thousand U.S. Dollars ($50,000.00). Prizes are non-transferable and may not be substituted, except that Sponsor reserves the right to substitute a prize of equal or greater value if the advertised prize becomes unavailable.
Winners are solely responsible for all federal, state, local, and foreign taxes, if any, arising from receipt of a prize. U.S.-based winners will be required to provide a completed IRS Form W-9 prior to prize disbursement. Non-U.S. winners will be required to provide a completed IRS Form W-8BEN (or the appropriate W-8 series form). Sponsor will issue an IRS Form 1099-MISC to U.S. winners as required by law. Failure to provide required tax documentation within thirty (30) days of request may result in forfeiture of the prize.
7. NOTIFICATION OF WINNERS
Winners will be notified via the email address from which their submission was sent. If a winner does not respond within fourteen (14) days of notification, the prize may be forfeited and, at Sponsor’s discretion, awarded to an alternate submission.
8. INTELLECTUAL PROPERTY
Participants retain ownership of their submissions. By entering the Contest, each participant grants Sponsor a non-exclusive, perpetual, irrevocable, worldwide, royalty-free license to use, reproduce, modify, publish, distribute, and create derivative works from the submission, in whole or in part, in any medium now known or hereafter devised, for any purpose, including without limitation research and publication.
Each participant represents and warrants that: (a) the submission is the participant’s own original work; (b) the participant has the right to grant the license described above; and (c) the submission does not infringe or misappropriate any third party’s intellectual property or other rights.
9. COLLABORATION AND PUBLIC DISCLOSURE
Collaboration among participants is permitted and encouraged. Participants are requested (but not required) to refrain from publicly publishing specific backdoor triggers or detailed solutions prior to April 1, 2026. General discussion of approaches and techniques is welcome, including on HuggingFace’s community forums, on Reddit, Bluesky, Mastodon, or X. After April 1, 2026, Sponsor encourages participants to publish their write-ups publicly.
10. NO EMPLOYMENT RELATIONSHIP
Participation in the Contest does not constitute an application for employment and does not create any employment, contractor, agency, or other professional relationship between any participant and Sponsor. Sponsor makes no promise, guarantee, or representation that any participant will be contacted regarding employment or any other opportunity. Any future engagement between Sponsor and a participant would be subject to a separate agreement and Sponsor’s standard hiring processes.
11. PRIVACY AND DATA
Sponsor collects the following information in connection with the Contest:
Submissions: Email address and written submission content, collected at time of entry.
API Usage: Requests submitted through the hosted API may generate transient logs. Sponsor does not intentionally retain these logs.
Payment Information: Collected from winners only, solely for the purpose of prize fulfillment.
Personal information will be handled in accordance with Sponsor’s Privacy Policy, available at https://www.janestreet.com/privacy-notice/. Participants who are residents of the European Economic Area, United Kingdom, or Switzerland should be aware that their personal data may be transferred to and processed in the United States. By entering the Contest, such participants consent to this transfer.
12. LIMITATION OF LIABILITY AND INDEMNIFICATION
To the fullest extent permitted by applicable law, Sponsor and its officers, directors, employees, agents, and affiliates (collectively, the “Released Parties”) shall not be liable for: (a) any injury, loss, or damage of any kind arising from participation in the Contest or from the acceptance, possession, or use of any prize; (b) technical failures, including but not limited to malfunctions of the API, hosting infrastructure, or any computer, network, or electronic system; or (c) any errors in the administration of the Contest or the processing of submissions.
By entering the Contest, each participant agrees to release, discharge, and hold harmless the Released Parties from any and all claims, damages, or liabilities arising out of or related to participation in the Contest, including any claim based on publicity rights, defamation, or invasion of privacy.
Each participant agrees to indemnify and hold harmless the Released Parties from any claims arising from the participant’s breach of these Official Rules or from the participant’s submission, including any claim of intellectual property infringement.
13. GENERAL CONDITIONS
Sponsor reserves the right, in its sole discretion, to modify, suspend, or terminate the Contest at any time for any reason, including but not limited to fraud, technical failures, or any other factor that impairs the integrity of the Contest. In the event of termination, Sponsor may, in its discretion, award prizes based on submissions received prior to termination.
Any attempt to deliberately undermine the legitimate operation of the Contest, including but not limited to tampering with the hosted models or infrastructure, may result in disqualification and, if applicable, legal action.
By entering, participants agree to be bound by these Official Rules and the decisions of the Judging Panel, which are final and binding in all respects.
14. GOVERNING LAW AND DISPUTES
These Official Rules and any disputes arising out of or related to the Contest shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of law provisions. Any dispute that cannot be resolved informally shall be subject to the exclusive jurisdiction of the state and federal courts located in Manhattan, New York City, New York, USA.
15. WINNERS LIST
A list of winners will be available after prize fulfillment is complete. The list of winners will be published on the Contest website.  


