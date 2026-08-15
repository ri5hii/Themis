#!/usr/bin/env python3
"""Deterministic synthetic lease-corpus generator (recreates the Claude spec).

DEPRECATED (experimental, keep for reproduction only): this builds the v0.2
supervised-training corpus, the line learning.md's "training chain is obsolete"
decision removed from the shipped product path (deterministic engine +
grammar-constrained SLM for prose only). Not wired into the engine; do not
resurrect as the source of fallback training data.

Reproduces the three synthetic datasets the prompt defined — original, varied
lease language (never reproduces real contracts or the Leivaditi corpus):

  - deontic_multilabel : sentence-level deontic classification, ~250 rows per
    category (obl/ent/pro/per/oth/nen/none) → 1,750 rows, party ~50/50.
  - redflag_paragraph  : paragraph-level red-flag classification, 390 rows with
    the over-sampling distribution (3x30, 6x25, 11x10, none 40).
  - deontic_span       : sentence-level red-flag spans, 9 rare x8 + 11 common
    x4 = 116 rows.

Output is written to a *new* directory (default data/synthetic_generated/) and
validated against the real task vocab + engine schema using the same checks as
scripts/ingest_synthetic.py. Existing data/synthetic/ is never touched.

Usage:
    python scripts/build_synthetic.py [--seed 42] [--out data/synthetic_generated]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEONTIC_ORDER = ["obl", "ent", "pro", "per", "oth", "nen", "none"]
DEONTIC_LABEL = {c: i for i, c in enumerate(DEONTIC_ORDER)}
DEONTIC_PER_CATEGORY = 250

# redflag class -> target paragraph count (the prompt's over-sampling spec).
REDFLAG_COUNTS: dict[str, int] = {
    "riders": 30,
    "no_obligation_to_operate": 30,
    "assignment_indeplaatsstelling_permitted": 30,
    "additional_remarks": 25,
    "expansion": 25,
    "extension_period": 25,
    "special_stipulations": 25,
    "compalsory_reconstraction": 25,
    "warrantees_of_the_owner": 25,
    "break_option": 10,
    "change_of_control": 10,
    "damage": 10,
    "guarantee_transferable": 10,
    "holdover": 10,
    "landlord_repairs": 10,
    "reinstatement_clause": 10,
    "right_of_first_refusal_to_lease": 10,
    "right_of_first_refusal_to_purchase": 10,
    "services_charges": 10,
    "sublease_permitted": 10,
    "none": 40,
}

# Rare classes: 8 span rows each; the rest 4 each (total 116).
SPAN_RARE = [
    "riders",
    "no_obligation_to_operate",
    "assignment_indeplaatsstelling_permitted",
    "additional_remarks",
    "expansion",
    "extension_period",
    "special_stipulations",
    "compalsory_reconstraction",
    "warrantees_of_the_owner",
]

# Party label pools (landlord, tenant) — incl. real-ish company names.
PARTIES: list[tuple[str, str]] = [
    ("Landlord", "Tenant"),
    ("Lessor", "Lessee"),
    ("Party A", "Party B"),
    ("Kensington Realty LLC", "Brightline Logistics Inc."),
    ("Harborview Property Group", "Millbrook Distribution Co."),
    ("Alder Gate Holdings", "Summit Office Partners"),
    ("Northgate REIT", "Cloverline Analytics Corp."),
    ("Redwood Industrial Trust", "Keystone Freight Systems"),
    ("Owner", "Occupant"),
    ("Prospect Street Partners", "Cedarline Retail Group"),
]

CLAUSE_NUMS = [
    "Section 4.1", "Section 4.2", "Section 8(a)", "Section 12.2",
    "Section 5(b)", "Article 3", "Article 7", "Clause 9",
    "Paragraph 14.2", "Section 17.1", "Section 21.3", "Paragraph 3(c)",
]
AMOUNTS = [
    "Five Thousand Dollars ($5,000)", "$3,250", "One Thousand Two Hundred Dollars ($1,200)",
    "the sum of $8,400 per annum", "seventy-five percent (75%)", "$2,100",
    "One Hundred Fifty Dollars ($150)", "the amount of $4,900", "$750",
    "twelve percent (12%) of the annual Base Rent",
]
DAYS = [5, 10, 15, 20, 30, 45, 60, 90]
YEARS = [1, 3, 5, 10]


def _pools(variant: int) -> tuple[str, str, str, int, int, str]:
    """Deterministic slot values for a given (template, variant) pair."""
    l, t = PARTIES[variant % len(PARTIES)]
    num = CLAUSE_NUMS[(variant // len(PARTIES)) % len(CLAUSE_NUMS)]
    amt = AMOUNTS[(variant // (len(PARTIES) * len(CLAUSE_NUMS))) % len(AMOUNTS)]
    days = DAYS[(variant // (len(PARTIES) * len(CLAUSE_NUMS) * len(AMOUNTS))) % len(DAYS)]
    years = YEARS[(variant // (len(PARTIES) * len(CLAUSE_NUMS) * len(AMOUNTS) * len(DAYS))) % len(YEARS)]
    return l, t, num, days, years, amt


def _render(template: str, variant: int) -> str:
    l, t, num, days, years, amt = _pools(variant)
    return template.format(L=l, T=t, num=num, days=days, years=years, amt=amt)


# Deontic sentence templates per category. Text reads naturally for every
# party-naming pool (Landlord/Tenant, Lessor/Lessee, Party A/B, company names).
DEONTIC_TEMPLATES: dict[str, list[str]] = {
    "obl": [
        "{T} shall pay to {L} the Base Rent set forth in {num} on or before the first day of each calendar month.",
        "{L} shall maintain the structural elements of the Premises in good working order throughout the Lease Term.",
        "{T} shall provide {L} with written notice of any planned alterations at least {days} days before commencement.",
        "{L} shall reimburse {T} for approved Tenant Improvements within {days} days of receipt of a proper invoice.",
        "{T} shall keep the interior of the Premises clean and free of debris at all times.",
        "{L} shall deliver possession of the Premises to {T} on the Commencement Date in broom-clean condition.",
        "{T} shall comply with all applicable laws, codes, and regulations governing the use of the Premises.",
        "{L} shall provide {T} with access to the Premises during normal business hours for installation of fixtures.",
        "{T} shall obtain and maintain the insurance coverages required by {num} at its sole cost and expense.",
        "{L} shall cause the Building systems to be operated in accordance with Prudent Standards of Practice.",
        "{T} shall pay the Security Deposit to {L} concurrently with the execution of this Lease.",
        "{L} shall restore the Premises to their original condition at the expiration of the Lease Term.",
        "{T} shall give {L} not less than {days} days' prior written notice of any transfer of the Lease.",
        "{L} shall make all repairs to the roof and exterior walls of the Building at its own expense.",
        "{T} shall not use the Premises for any purpose other than general office use without {L}'s consent.",
        "{L} shall keep the common areas in a clean and safe condition for the benefit of all tenants.",
        "{T} shall furnish {L} with a current certificate of insurance evidencing the required coverage.",
        "{L} shall give {T} prior written notice before entering the Premises except in the case of emergency.",
        "{T} shall pay all charges for utilities separately metered to the Premises within {days} days of billing.",
        "{L} shall pay for all repairs to the Building's mechanical systems serving the Premises.",
        "{T} shall replace any damaged windows or glass in the Premises at its own expense.",
        "{L} shall provide {T} with quiet enjoyment of the Premises during the Lease Term.",
        "{T} shall advise {L} in writing of any intended sublease at least {days} days in advance.",
        "{L} shall maintain the parking areas serving the Premises in good order and adequate lighting.",
        "{T} shall be responsible for the disposal of all trash and refuse generated in the Premises.",
        "{L} shall return the Security Deposit, less permitted deductions, within {days} days of Lease expiration.",
        "{T} shall pay any late fee described in {num} within {days} days of the applicable notice.",
        "{L} shall provide {T} with a copy of any notice of default served upon {L} under the Building loan.",
        "{T} shall take good care of the Premises and shall not commit or suffer waste or damage.",
        "{L} shall pay all real property taxes assessed against the Building for the term of this Lease.",
        "{T} shall install and maintain smoke detectors in the Premises in accordance with applicable law.",
        "{L} shall give {T} at least {days} days' notice of any intended sale of the Building.",
        "{T} shall deliver a signed estoppel certificate to {L} within {days} days of {L}'s request.",
        "{L} shall keep the public corridors and elevator lobbies in a neat and orderly appearance.",
        "{T} shall procure and maintain general liability insurance with limits of at least {amt}.",
        "{L} shall pay for all costs of capital improvements required to make the Building compliant with law.",
        "{T} shall submit any plans for Tenant Improvements to {L} for approval prior to construction.",
        "{L} shall advise {T} of any proposed change in the Building operating hours.",
        "{T} shall preserve the confidentiality of any proprietary information shared by {L}.",
        "{L} shall re-key the locks to the Premises upon {T}'s request at {T}'s cost.",
    ],
    "ent": [
        "{T} shall have the right to renew this Lease for an additional term of {years} years upon written notice.",
        "{L} shall be entitled to terminate this Lease in the event of an uncured Event of Default.",
        "{T} shall have the option to expand the Premises by taking additional space on the same floor.",
        "{L} shall have the right to relocate {T} to comparable space elsewhere in the Building upon {days} days' notice.",
        "{T} shall be entitled to a rent abatement of {days} days if the Premises are rendered untenantable.",
        "{L} shall have the right to approve any signage proposed by {T} on the exterior of the Building.",
        "{T} shall have the right to sublease the Premises with the prior written consent of {L}, which consent shall not be unreasonably withheld.",
        "{L} shall be entitled to enter the Premises at reasonable times to make repairs and inspect the same.",
        "{T} shall have the option to purchase the Building if {L} elects to sell during the Lease Term.",
        "{L} shall have the right to terminate this Lease if {T} abandons the Premises for {days} consecutive days.",
        "{T} shall be entitled to install a satellite dish on the roof subject to {L}'s approval.",
        "{L} shall have the right to change the Building name or the street address of the Premises.",
        "{T} shall have the right to use the Building conference center upon advance reservation.",
        "{L} shall be entitled to require {T} to provide additional security deposits in the event of default.",
        "{T} shall have the option to extend the Lease Term for {years} years at the then-prevailing market rent.",
        "{L} shall have the right to consent to or withhold consent to any assignment in its reasonable discretion.",
        "{T} shall be entitled to install its own lighting fixtures in the Premises at its expense.",
        "{L} shall have the right to designate the providers of janitorial and other building services.",
        "{T} shall have the right to an annual inspection of the Building's fire and life safety systems.",
        "{L} shall be entitled to terminate the Lease upon the demolition or condemnation of the Building.",
        "{T} shall have the right to place the name of its business on the Building directory.",
        "{L} shall have the right to require {T} to remove any Alterations at the end of the Lease Term.",
        "{T} shall be entitled to a full credit for any unamortized Tenant Improvements upon early termination.",
        "{L} shall have the right to limit the hours of access to the Premises for security purposes.",
        "{T} shall have the option to install an HVAC unit serving the Premises with {L}'s consent.",
        "{L} shall be entitled to withhold consent to a transfer if the assignee lacks reasonable net worth.",
        "{T} shall have the right to terminate the Lease if the Building fails to comply with ADA requirements.",
        "{L} shall have the right to close the Building on recognized holidays and other customary dates.",
        "{T} shall be entitled to a credit against rent for any operating expense savings achieved.",
        "{L} shall have the right to grant easements and other rights over the Building to third parties.",
        "{T} shall have the right to request a subordination of this Lease to any Building financing.",
        "{L} shall be entitled to audit {T}'s books to verify percentage rent calculations.",
        "{T} shall have the option to install a backup generator serving the Premises at its cost.",
        "{L} shall have the right to retain {T}'s Security Deposit to cure any monetary default.",
        "{T} shall be entitled to the exclusive use of the freight elevator during move-in.",
        "{L} shall have the right to require {T} to comply with the Building's security rules.",
        "{T} shall have the right to display its logo at the entrance to the Premises.",
        "{L} shall be entitled to withhold consent to a transfer that would reduce the Building's credit.",
        "{T} shall have the option to convert this Lease to a full-service lease upon the terms agreed.",
        "{L} shall have the right to install telecommunication equipment on the roof of the Building.",
    ],
    "pro": [
        "{T} shall not assign or sublet the Premises without the prior written consent of {L}.",
        "{L} shall not unreasonably withhold, condition, or delay its consent to a proposed transfer.",
        "{T} shall not use the Premises for any use that violates the Certificate of Occupancy.",
        "{L} shall not terminate this Lease except for the causes expressly set forth in {num}.",
        "{T} shall not make any Alterations without first obtaining {L}'s written approval.",
        "{L} shall not raise the Base Rent more frequently than once every twelve months.",
        "{T} shall not keep any hazardous materials in the Premises in violation of applicable law.",
        "{L} shall not enter the Premises during the Lease Term except at reasonable hours and with notice.",
        "{T} shall not store goods or materials in the common corridors or public areas of the Building.",
        "{L} shall not unreasonably withhold its consent to {T}'s proposed signage.",
        "{T} shall not make any claim or lien against the Building or the Premises for labor or materials.",
        "{L} shall not disturb {T}'s quiet enjoyment of the Premises during the Lease Term.",
        "{T} shall not smoke or permit smoking within the Premises or the common areas of the Building.",
        "{L} shall not require {T} to pay any sum for repairs to the Premises occasioned by ordinary wear and tear.",
        "{T} shall not exceed the floor load or use the Premises for any extraordinary or dangerous use.",
        "{L} shall not change the ratio of the Premises to the total rentable area of the Building arbitrarily.",
        "{T} shall not install any equipment that emits excessive vibration or noise without {L}'s consent.",
        "{L} shall not fail or refuse to provide the services required of it under {num}.",
        "{T} shall not place any sign on the exterior of the Building without the prior written consent of {L}.",
        "{L} shall not require {T} to pay additional rent based on a change in the method of measuring the Premises.",
        "{T} shall not dispose of any materials in the Premises in violation of environmental laws.",
        "{L} shall not withhold consent to an assignment to a financially responsible and reputable assignee.",
        "{T} shall not create any lien, mortgage, or encumbrance against the Premises.",
        "{L} shall not reduce the amount of parking available to {T} during the Lease Term.",
        "{T} shall not conduct any auction, sale, or public exhibition in the Premises without consent.",
        "{L} shall not terminate the Lease for a non-monetary default until the applicable notice period has expired.",
        "{T} shall not use the Premises for the storage of any flammable or explosive substances.",
        "{L} shall not unreasonably withhold consent to {T}'s installation of telecommunication equipment.",
        "{T} shall not allow the Premises to be used by any person other than in the ordinary course of business.",
        "{L} shall not permit the Building to fall below the standards required by the applicable codes.",
        "{T} shall not alter the internal plumbing or electrical systems of the Premises without consent.",
        "{L} shall not charge {T} for utility services already included in the fixed rent.",
        "{T} shall not post any notices or advertisements on the exterior of the Premises.",
        "{L} shall not require {T} to accept a form of consent that is not consistent with industry practice.",
        "{T} shall not keep animals in the Premises except service animals permitted by law.",
        "{L} shall not impose any charge upon {T} for the mere approval of a permitted transfer.",
        "{T} shall not violate any rule of the Building promulgated by {L} from time to time.",
        "{L} shall not change the hours of operation of the Building without reasonable advance notice.",
        "{T} shall not subdivide the Premises into separate demised premises without {L}'s consent.",
        "{L} shall not discriminate among tenants in the provision of services or the allocation of costs.",
    ],
    "per": [
        "{T} may install furniture, partitions, and movable fixtures in the Premises at its expense.",
        "{L} may inspect the Premises upon reasonable advance notice and at reasonable hours.",
        "{T} may terminate this Lease upon the payment of a termination fee equal to {amt}.",
        "{L} may temporarily close the common areas of the Building for repairs or maintenance.",
        "{T} may display its business name on the Building directory without additional charge.",
        "{L} may change the method of allocating operating expenses in accordance with {num}.",
        "{T} may use the freight elevator at times designated by {L} for move-in and move-out.",
        "{L} may sell, assign, or encumber the Building subject to the terms of this Lease.",
        "{T} may make cosmetic changes to the Premises that do not affect the structure of the Building.",
        "{L} may require {T} to participate in the Building's recycling program.",
        "{T} may renew this Lease by giving {L} written notice not later than {days} days before expiration.",
        "{L} may impose reasonable rules for the use of the Building's loading docks.",
        "{T} may install temporary partitions to divide the Premises for meetings and storage.",
        "{L} may retain qualified consultants to verify the accuracy of operating expense statements.",
        "{T} may use the rooftop deck during normal business hours subject to the Building rules.",
        "{L} may terminate this Lease if {T} fails to cure a default within the applicable cure period.",
        "{T} may request that {L} provide after-hours HVAC service at rates set forth in {num}.",
        "{L} may enter the Premises in the event of an emergency without prior notice.",
        "{T} may assign this Lease to an affiliate upon notice to {L}, provided the assignee meets certain criteria.",
        "{L} may relocate the Building's mail room or package center to another portion of the Building.",
        "{T} may purchase energy from third-party providers where permitted by applicable law.",
        "{L} may suspend access to the Premises during a force majeure event.",
        "{T} may conduct a permitted move-in or move-out at any time upon reasonable scheduling.",
        "{L} may adjust the Base Rent for additional space taken by {T} during the Lease Term.",
        "{T} may install secure keycard access systems within the Premises at its own cost.",
        "{L} may require {T} to keep the Premises open for business during normal retail hours.",
        "{T} may terminate the Lease upon {days} days' notice if the Premises are destroyed.",
        "{L} may enter the Premises to make repairs required to be made by {T} after {days} days' notice.",
        "{T} may use the Premises for general office, storage, and related uses approved by {L}.",
        "{L} may approve or disapprove {T}'s proposed Alterations in its reasonable discretion.",
        "{T} may install energy-efficient lighting in the Premises with {L}'s consent.",
        "{L} may request annual audited financial statements from {T} if {T} is a corporation.",
        "{T} may use common meeting rooms on a first-come, first-served basis.",
        "{L} may require {T} to remove any Alterations made without the required consent.",
        "{T} may advertise its location in the Premises in any medium it deems appropriate.",
        "{L} may designate after-hours entry procedures that {T} must follow.",
        "{T} may install a small satellite or antenna on the roof subject to {L}'s approval.",
        "{L} may adopt and modify Building rules from time to time in a reasonable manner.",
        "{T} may withhold rent if {L} fails to provide essential services after written notice.",
        "{L} may require {T} to provide a certificate of insurance within {days} days of a request.",
    ],
    "oth": [
        "This Lease is subject and subordinate to all ground leases, mortgages, and other liens affecting the Building.",
        "Time is of the essence with respect to each and every obligation of {T} under this Lease.",
        "This Lease shall be governed by and construed in accordance with the laws of the State of Delaware.",
        "Any notice required under this Lease shall be in writing and delivered in accordance with {num}.",
        "The captions and headings in this Lease are for convenience only and shall not affect its interpretation.",
        "This Lease constitutes the entire agreement between the parties and supersedes all prior negotiations.",
        "The obligations of the parties hereunder shall be binding upon and inure to their respective successors and assigns.",
        "In the event of any inconsistency between this Lease and any Exhibit, the terms of this Lease shall prevail.",
        "All monetary amounts under this Lease are stated and payable in lawful currency of the United States.",
        "The failure of either party to enforce any provision of this Lease shall not be deemed a waiver thereof.",
        "This Lease may be executed in counterparts, each of which shall be deemed an original.",
        "The parties waive trial by jury in any action arising out of or relating to this Lease.",
        "This Lease shall not be recorded without the prior written consent of {L}.",
        "The per diem value of any provision of this Lease shall be determined in accordance with {num}.",
        "Each party represents that it has full power and authority to enter into this Lease.",
        "The effective date of this Lease shall be the date of the last party to execute the same.",
        "All exhibits and riders attached to this Lease are incorporated herein by reference.",
        "Any provision of this Lease found invalid shall be severable, and the remainder shall remain in full force.",
        "The parties agree that the Premises are leased as is, without any representation or warranty by {L}.",
        "This Lease shall be construed without regard to the rule against perpetuities.",
        "The prevailing party in any dispute arising under this Lease shall be entitled to its reasonable attorneys' fees.",
        "Any amount not paid when due under this Lease shall bear interest at the rate set forth in {num}.",
        "The relationship of the parties is that of landlord and tenant and not that of partners or joint venturers.",
        "Each party waives any right to a jury trial and any right to recover consequential damages.",
        "This Lease may be amended only by a written instrument signed by both parties.",
        "The terms of this Lease shall be construed as covenants running with the land.",
        "Both parties acknowledge having read this Lease and having the opportunity to consult counsel.",
        "The obligations of {L} are subject to the terms of any Building mortgage existing at execution.",
        "Any consent required of a party under this Lease shall not be unreasonably withheld, conditioned, or delayed.",
        "The submission of this Lease for examination does not constitute an offer or an option to lease.",
        "This Lease is not intended to create any rights in any third-party beneficiary.",
        "The parties agree to negotiate in good faith with respect to any matter not addressed in this Lease.",
        "All rights and remedies of the parties are cumulative and not exclusive of any rights or remedies at law.",
        "This Lease shall be interpreted in accordance with its plain meaning and not against the drafter.",
        "The tenant is permitted to rely on the provisions of this Lease only as of its effective date.",
        "Any reference to a statute or regulation shall include any successor statute or regulation.",
        "The parties intend that the Premises be used solely for the uses permitted under {num}.",
        "No representation of any agent of {L} is binding unless confirmed in writing by {L}.",
        "This Lease shall terminate automatically upon the mutual written agreement of the parties.",
        "The parties agree that any modification of this Lease shall require an executed amendment.",
    ],
    "nen": [
        "{T} shall have no right to renew or extend this Lease upon the expiration of the initial term.",
        "{L} shall have no right to withhold its consent to a permitted assignment by {T}.",
        "{T} waives any right to claim that this Lease has been renewed by operation of law or estoppel.",
        "{L} disclaims any right to terminate this Lease for a default that is not an Event of Default.",
        "{T} shall have no right to withhold the Security Deposit paid by {T} to {L}.",
        "{L} shall have no right to charge {T} for standard services already covered by the Base Rent.",
        "{T} waives any right to offset its rent against amounts owed by {L} to {T}.",
        "{L} shall have no right to require {T} to maintain insurance in excess of the limits set forth in {num}.",
        "{T} shall have no right to use the Premises for any use not expressly permitted by {num}.",
        "{L} waives any right to terminate the Lease based on the mere passage of time.",
        "{T} shall have no right to assign this Lease to a competitor of {L}.",
        "{L} shall have no right to inspect {T}'s financial records except as expressly provided herein.",
        "{T} waives any right to a jury trial in any dispute arising under this Lease.",
        "{L} shall have no right to increase the Base Rent more than once per calendar year.",
        "{T} shall have no right to install signage visible from the street without {L}'s consent.",
        "{L} disclaims any right to collect double rent from {T} for the same period.",
        "{T} waives any right to recover consequential damages arising from a breach of this Lease.",
        "{L} shall have no right to enter the Premises without giving {T} prior written notice.",
        "{T} shall have no right to make any Alterations that affect the Building's structure.",
        "{L} shall have no right to terminate this Lease during the first {years} years of the term.",
        "{T} waives any right to terminate this Lease due to the condition of the Premises.",
        "{L} shall have no right to retain {T}'s Security Deposit without an itemized statement.",
        "{T} shall have no right to share the Premises with any other occupant without {L}'s consent.",
        "{L} disclaims any right to hold {T} liable for the acts of other tenants.",
        "{T} waives any right to a reduction in rent for the temporary interruption of services.",
        "{L} shall have no right to change the access routes to the Premises without reasonable notice.",
        "{T} shall have no right to park in the Building garage beyond the allotted spaces.",
        "{L} waives any right to accelerate rent upon a transfer by {T} to a permitted assignee.",
        "{T} shall have no right to request a subordination of this Lease during the initial term.",
        "{L} shall have no right to require {T} to replace items subject to normal wear and tear.",
        "{T} waives any right to the exclusive use of the common areas of the Building.",
        "{L} shall have no right to charge {T} for repairs occasioned by the negligence of {L}.",
        "{T} shall have no right to terminate this Lease for a mere violation of the Building rules.",
        "{L} disclaims any right to restrict {T}'s use of its own equipment within the Premises.",
        "{T} waives any right to claim a constructive eviction during a temporary interruption of access.",
        "{L} shall have no right to require {T} to pay a surcharge for late payment of a permitted amount.",
        "{T} shall have no right to install any permanent partitions without {L}'s written consent.",
        "{L} shall have no right to object to {T}'s reasonable trade fixtures.",
        "{T} waives any right to the return of any portion of the Security Deposit except as provided.",
        "{L} shall have no right to recover from {T} any unamortized costs of Alterations made by {L}.",
    ],
    "none": [
        "This Lease Agreement is made as of the date set forth on the first page hereof.",
        "The Premises are located at the address identified in the Basic Lease Information.",
        "The Base Rent for the initial Lease Year shall be the amount set forth in {num}.",
        "The Commencement Date is the earlier of the Lease Commencement Date and the date of substantial completion.",
        "The Security Deposit shall be held by {L} in a non-interest-bearing account.",
        "Notices to {T} shall be sent to the address of the Premises as first set forth above.",
        "The rentable area of the Premises is approximately the square footage stated in the Data Sheet.",
        "The Lease Term shall commence on the Commencement Date and expire on the Expiration Date.",
        "The parties have executed this Lease in several counterparts, each of which is an original.",
        "The Building is located on the parcel described in the legal description attached as Exhibit A.",
        "The Broker for this transaction is identified in the Basic Lease Information.",
        "The Premises are delivered with the existing fixtures and equipment as shown on the floor plan.",
        "The Lease shall be effective upon the execution and delivery of this document by both parties.",
        "The address for notices to {L} is set forth in the first paragraph of this Lease.",
        "The parties agree that the data set forth in the Basic Lease Information is accurate as of execution.",
        "This instrument is executed by the parties as of the date first above written.",
        "The exhibits listed on the cover page are attached hereto and made a part hereof.",
        "The parties have initialed each page of this Lease as evidence of their agreement.",
        "The Basic Lease Information is incorporated into this Lease by this reference.",
        "The Premises are in the Building commonly known by the address set forth above.",
        "The Lease Term is set forth in the Basic Lease Information and shall not be modified except by amendment.",
        "The parties hereby acknowledge receipt of a copy of this fully executed Lease.",
        "The square footage of the Premises is measured to the interior faces of the perimeter walls.",
        "This Lease is being entered into by the parties in connection with the transaction described above.",
        "The parties have attached the applicable exhibits and schedules to this Lease.",
        "The Rent Commencement Date shall be the date set forth in the Basic Lease Information.",
        "Each party has retained counsel of its choice in connection with the negotiation of this Lease.",
        "The Premises are deemed delivered when the premises are ready for {T}'s occupancy.",
        "The exhibits to this Lease are integral parts of this Lease and are binding on the parties.",
        "The parties agree that the effective date is the date of delivery of the last executed counterpart.",
        "The Property is zoned for the uses set forth in the municipal zoning ordinance.",
        "The executed Lease and its exhibits constitute the complete agreement between the parties.",
        "The parties have agreed upon the amounts and rates set forth in the Basic Lease Information.",
        "The security deposit required under this Lease is identified in the first paragraph.",
        "The Building contains the number of stories and parking spaces described in the Data Sheet.",
        "The parties waive any requirement that this Lease be delivered by notarized instrument.",
        "The Lease is being negotiated on an arm's-length basis between the parties.",
        "The parties acknowledge that the market conditions as of the date hereof are reflected in the rent.",
        "The parties have exchanged drafts of this Lease prior to final execution.",
        "The date on which {T} takes possession of the Premises shall be set forth in the Commencement Certificate.",
    ],
}

# Red-flag paragraph templates per class (full paragraph, 1-4 sentences).
REDFLAG_TEMPLATES: dict[str, list[str]] = {
    "riders": [
        "Rider No. 1, attached hereto and made a part of this Lease, sets forth additional provisions regarding the Premises and shall control over any conflicting provision in the body of this Lease.",
        "The parties have executed Rider A, which modifies the terms of this Lease to permit the installation of a backup generator.",
        "A rider attached as Exhibit N to this Lease addresses the allocation of parking spaces between the parties.",
        "The Special Stipulations Rider attached hereto grants {T} a one-time right to terminate this Lease at the end of the fifth Lease Year.",
        "Rider No. 2 changes the permitted use of the Premises from general office to research and development.",
        "The parties acknowledge that the rider attached as Schedule 3 sets forth the agreed terms for the Tenant Improvement allowance.",
        "Rider B attached to this Lease confirms that the Base Rent shall be abated for the first three months of the Lease Term.",
        "The parties have agreed to the terms set forth in the riders attached to this Lease, which are incorporated herein by reference.",
        "A rider to this Lease grants {T} the right of first refusal with respect to the expansion premises on the seventh floor.",
        "The rider attached as Exhibit R addresses the obligation of {L} to provide after-hours HVAC service to the Premises.",
        "The parties have attached a rider concerning the maintenance of the Premises' fire suppression system.",
        "Rider No. 3 provides that the Security Deposit shall be reduced to one month's Base Rent upon the fifth anniversary.",
        "A rider to this Lease sets forth the terms under which {T} may assign the Lease to a permitted affiliate.",
        "The riders attached to this Lease have been initialed by the parties and shall prevail over the printed form.",
        "Rider C confirms the parties' agreement regarding the delivery of the Premises in a turnkey condition.",
        "The parties have added a rider governing the use of the Building's rooftop for {T}'s telecommunications equipment.",
        "A rider to this Lease addresses the responsibility for window washing and exterior maintenance of the Premises.",
        "The parties have agreed that the rider attached as Exhibit X shall govern the allocation of escalations.",
        "Rider No. 4 modifies the insurance requirements set forth in Section 9 of this Lease.",
        "A rider attached to this Lease sets forth the terms under which {T} may install an internal stair between floors.",
        "The parties have executed a rider confirming the amount and timing of the Tenant Improvement allowance.",
        "Rider D addresses the signage rights of {T} on the exterior of the Building.",
        "A rider to this Lease provides that {T} shall have access to the Premises twenty-four hours per day, seven days per week.",
        "The parties have attached a rider setting forth the agreed schedule for the payment of estimated operating expenses.",
        "Rider No. 5 confirms that the Premises shall be used for a medical office and shall comply with applicable healthcare regulations.",
        "A rider to this Lease grants {T} the option to purchase the Building subject to the terms set forth therein.",
        "The parties have agreed that the rider attached as Exhibit B shall control over any conflicting provision herein.",
        "Rider E modifies the provisions of Section 14 to reduce the required notice period for a permitted assignment.",
        "A rider attached to this Lease addresses the installation and removal of a vault and floor reinforcement.",
        "The parties have executed a rider that supersedes the printed provisions relating to the late payment of rent.",
    ],
    "no_obligation_to_operate": [
        "{L} shall have no obligation to operate the Building as a first-class office building during the Lease Term.",
        "The parties acknowledge that {L} is not obligated to provide any services to the Premises other than those expressly set forth herein.",
        "{L} shall have no obligation to lease or rent any other space in the Building to any particular tenant.",
        "{T} acknowledges that {L} has no obligation to construct, improve, or alter the Premises prior to delivery.",
        "{L} shall have no obligation to maintain the Building beyond the standards required by applicable law.",
        "The parties agree that {L} shall have no obligation to operate the Building's retail concourse.",
        "{L} shall have no obligation to provide continuous operation of the Building systems.",
        "{T} acknowledges that {L} is under no obligation to renew any service contract for the Building's equipment.",
        "{L} shall have no obligation to keep the common areas open or accessible at any particular hours.",
        "The parties agree that {L} shall have no obligation to promote or market the Premises or the Building.",
        "{L} shall have no obligation to expand, upgrade, or improve the Building's parking facilities.",
        "{T} acknowledges that {L} has no obligation to relocate {T} to other premises in the Building.",
        "{L} shall have no obligation to maintain the temperature of the Premises within any particular range.",
        "The parties agree that {L} shall have no obligation to provide security services for the Premises.",
        "{L} shall have no obligation to repair or maintain the equipment installed by {T} in the Premises.",
        "{T} acknowledges that {L} is not obligated to provide any janitorial services to the Premises.",
        "{L} shall have no obligation to remove or dispose of any materials or waste generated by {T}.",
        "The parties agree that {L} shall have no obligation to insure any of {T}'s property.",
        "{L} shall have no obligation to provide any utility services to the Premises beyond the point of the Building's meters.",
        "{T} acknowledges that {L} has no obligation to install any additional plumbing or electrical capacity.",
        "{L} shall have no obligation to operate the elevators serving the Premises during weekends or holidays.",
        "The parties agree that {L} shall have no obligation to re-lease the Premises after the expiration of this Lease.",
        "{L} shall have no obligation to purchase any goods or services from {T} in connection with this Lease.",
        "{T} acknowledges that {L} is not obligated to make any repairs to the Premises during the Lease Term.",
        "{L} shall have no obligation to provide additional parking spaces for {T}'s employees or visitors.",
        "The parties agree that {L} shall have no obligation to comply with the Americans with Disabilities Act beyond the scope required by law.",
        "{L} shall have no obligation to maintain the interior finishes of the Premises after delivery.",
        "{T} acknowledges that {L} has no obligation to restore the Premises upon the expiration of the Lease Term.",
        "{L} shall have no obligation to install, operate, or maintain any telephone or data systems in the Premises.",
        "The parties agree that {L} shall have no obligation to grant any rights to {T} other than those set forth in this Lease.",
    ],
    "assignment_indeplaatsstelling_permitted": [
        "{T} may assign this Lease or sublet the whole or any part of the Premises without the consent of {L}.",
        "Assignment of this Lease shall be permitted upon written notice to {L}, without the requirement of consent.",
        "The parties agree that {T} may freely assign this Lease to any entity controlling, controlled by, or under common control with {T}.",
        "{T} shall be permitted to assign this Lease to a successor in connection with a merger or consolidation.",
        "The parties agree that any assignment or sublease of the Premises shall be permitted without the prior consent of {L}.",
        "{T} may sublet the Premises to any third party without the consent of {L}, provided that {L} is given prior written notice.",
        "Notwithstanding anything to the contrary, {T} shall be free to assign this Lease upon the sale of its business.",
        "The parties agree that {T} may transfer this Lease to any affiliate without the consent of {L}.",
        "{T} may assign this Lease without {L}'s consent if {T}'s net worth exceeds {amt}.",
        "The parties agree that {T} is free to sublet the Premises in whole or in part upon reasonable notice to {L}.",
        "Assignment and subletting of the Premises shall be unrestricted, subject only to written notice to {L}.",
        "The parties agree that {T} may assign this Lease in connection with a public offering of its securities.",
        "{T} shall be permitted to assign this Lease to any person who acquires all or substantially all of {T}'s assets.",
        "The parties agree that {T} may sublet the Premises to a tenant approved by {L} within a reasonable period of time.",
        "Assignment of this Lease by {T} shall not require the consent of {L} after the initial {years} years of the term.",
        "The parties agree that {T} may assign this Lease to a parent, subsidiary, or other entity under common ownership.",
        "{T} shall be free to assign this Lease upon the transfer of all of its issued and outstanding capital stock.",
        "The parties agree that any permitted transfer of this Lease shall not release {T} from its obligations hereunder.",
        "{T} may sublet the Premises to any of its employees, officers, or directors without the consent of {L}.",
        "The parties agree that {T} shall be permitted to assign this Lease to a bona fide purchaser of its business.",
        "{T} may assign this Lease without {L}'s consent provided that the assignee assumes the obligations of {T} in writing.",
        "The parties agree that assignment of this Lease shall be permitted upon the satisfaction of the conditions set forth in {num}.",
        "{T} shall be permitted to sublet any portion of the Premises that it does not currently occupy.",
        "The parties agree that {T} may assign this Lease to a joint venture in which {T} participates.",
        "Assignment of this Lease shall be permitted if {T} provides {L} with evidence of the assignee's financial responsibility.",
        "The parties agree that {T} may transfer this Lease to any corporation into which {T} is merged.",
        "{T} shall be permitted to assign this Lease to a receiver or trustee appointed in a bankruptcy proceeding.",
        "The parties agree that {T} may sublet the Premises for any use permitted under this Lease.",
        "{T} may assign this Lease to any person without the consent of {L}, and {L} waives any right of approval.",
        "The parties agree that {T} shall be free to assign this Lease at any time during the Lease Term.",
    ],
    "additional_remarks": [
        "The parties have agreed that the Base Rent shall be abated during the period the Premises are being prepared for occupancy.",
        "Additional remarks regarding the delivery of the Premises are set forth in the Basic Lease Information.",
        "The parties acknowledge that {T} has inspected the Premises and accepts the same in their current condition.",
        "The parties agree that the Security Deposit may be applied to cure any monetary default by {T}.",
        "Additional provisions regarding the use of the roof are set forth in the addendum attached hereto.",
        "The parties have agreed that the Building's janitorial services shall be provided on business days only.",
        "The parties acknowledge that certain improvements to the Premises remain to be completed by {L}.",
        "Additional remarks relating to the operation of the Building's heating and cooling systems are attached hereto.",
        "The parties agree that {T} shall provide {L} with a certificate of occupancy prior to occupying the Premises.",
        "The parties have noted that the Premises include a storage area in the basement of the Building.",
        "Additional remarks concerning the allocation of the cost of the Building's concierge services are set forth herein.",
        "The parties acknowledge that the Building's loading dock is shared among all tenants.",
        "The parties agree that {T} shall have access to the Building's conference facilities upon advance reservation.",
        "Additional remarks regarding the installation of window treatments in the Premises are attached.",
        "The parties have agreed that the signage requirements for the Premises are set forth in the signage plan.",
        "The parties acknowledge that the Building is currently undergoing renovations that may affect common areas.",
        "Additional remarks regarding the assignment of parking spaces to {T} are set forth in the parking plan.",
        "The parties agree that the Premises shall be delivered with the existing floor coverings intact.",
        "The parties have noted that the Building's security system requires an access card for entry after hours.",
        "Additional remarks relating to the annual escalation of the Base Rent are set forth in {num}.",
        "The parties agree that {T} shall cooperate with {L} in the installation of a new Building directory.",
        "The parties acknowledge that the Premises are subject to the covenants of the applicable condominium declaration.",
        "Additional remarks regarding the maintenance of the Premises' restroom facilities are attached.",
        "The parties have agreed that the effective date of the Lease shall be the later of the dates of execution.",
        "Additional remarks concerning the Building's compliance with sustainability standards are set forth herein.",
    ],
    "expansion": [
        "{T} shall have the right to expand the Premises by leasing additional space on the same floor of the Building.",
        "The parties agree that {T} may lease the expansion premises described in {num} upon the terms set forth therein.",
        "{T} shall have the option to expand into the adjoining premises when the current occupant vacates.",
        "The parties agree that {T} shall have the right of first refusal with respect to the expansion space on the floor.",
        "{T} shall have the right to expand the Premises into the additional area identified on the floor plan.",
        "The parties agree that the expansion space shall be delivered to {T} in its then-existing condition.",
        "{T} shall have the option to expand the Premises by up to an additional {years} square feet.",
        "The parties agree that {T} may expand the Premises upon {days} days' written notice to {L}.",
        "{T} shall have the right to expand into the space currently occupied by the adjacent tenant.",
        "The parties agree that the expansion of the Premises shall be at the then-prevailing market rent.",
        "{T} shall have the right to expand the Premises into the vacant space on the second floor.",
        "The parties agree that any expansion of the Premises shall be subject to the Building's structural capacity.",
        "{T} shall have the option to expand the Premises at any time during the Lease Term.",
        "The parties agree that {T} shall have a right of expansion with respect to the space across the hall.",
        "{T} shall have the right to expand the Premises upon the expiration of the term of the adjacent tenant.",
        "The parties agree that the expansion space shall be demised to {T} on the terms set forth in the expansion agreement.",
        "{T} shall have the option to expand the Premises to include the storage space in the basement.",
        "The parties agree that {T} may expand the Premises by terminating its obligations with respect to part of the space.",
        "{T} shall have the right to expand the Premises into the space on the mezzanine level.",
        "The parties agree that any expansion of the Premises shall not require a new Lease.",
        "{T} shall have the option to expand the Premises when the Building's anchor tenant relocates.",
        "The parties agree that {T} shall have a right to expand into the space designated as expansion premises.",
        "{T} shall have the right to expand the Premises by leasing the additional space on the floor plan marked Exhibit G.",
        "The parties agree that the expansion premises shall be offered to {T} before any third party.",
        "{T} shall have the right to expand the Premises into the adjoining premises upon the terms agreed by the parties.",
    ],
    "extension_period": [
        "{T} shall have the option to extend this Lease for an additional period of {years} years upon written notice.",
        "The parties agree that {T} may extend the Lease Term for one or more extension periods as set forth in {num}.",
        "{T} shall have the right to renew this Lease for an extension period at the then-prevailing fair market rent.",
        "The parties agree that the extension option may be exercised by {T} by notice delivered at least {days} days before expiration.",
        "{T} shall have the option to extend this Lease for a period of five years upon the terms and conditions set forth herein.",
        "The parties agree that the Lease Term shall be extended automatically unless {T} gives notice of non-extension.",
        "{T} shall have the right to extend this Lease for a renewal term on the same terms and conditions as the initial term.",
        "The parties agree that {T} may exercise the extension option only if no Event of Default has occurred.",
        "{T} shall have the option to extend the Lease Term for an additional period of {years} years at the then-prevailing market rate.",
        "The parties agree that the extension of this Lease shall not require the execution of a new lease document.",
        "{T} shall have the right to renew this Lease for an extension period upon written notice to {L}.",
        "The parties agree that {T} shall have a second option to extend this Lease for an additional period of {years} years.",
        "{T} shall have the option to extend the term of this Lease for a period of three years.",
        "The parties agree that the Base Rent during any extension period shall be determined in accordance with {num}.",
        "{T} shall have the right to extend this Lease for a term of {years} years commencing upon the expiration of the initial term.",
        "The parties agree that the extension option shall be exercisable only by {T} and not by any assignee of {T}.",
        "{T} shall have the option to extend this Lease for an additional period upon the terms set forth in the extension rider.",
        "The parties agree that {T} shall give {L} written notice of its election to extend not less than {days} days prior to expiration.",
        "{T} shall have the right to renew this Lease for an extension period of one year on an annual basis.",
        "The parties agree that the extension of this Lease shall be on a month-to-month basis after the initial term.",
        "{T} shall have the option to extend this Lease for a period of ten years subject to the Building's redevelopment rights.",
        "The parties agree that {T} shall have the right to extend this Lease for an additional period of {years} years.",
        "{T} shall have the option to renew this Lease for an extension period upon the terms set forth in the renewal rider.",
        "The parties agree that the extension option may be exercised at any time prior to the expiration of the Lease Term.",
        "{T} shall have the right to extend the Lease Term for a period equal to the initial term.",
    ],
    "special_stipulations": [
        "The Special Stipulations attached as Exhibit S set forth additional terms that are not contained in the printed form.",
        "The parties have agreed to the Special Stipulations set forth in Schedule 2, which supersede any conflicting provision.",
        "Special Stipulations regarding the operation of the Premises as a data center are attached hereto.",
        "The parties acknowledge the Special Stipulations regarding the installation of a redundant power supply for the Premises.",
        "The Special Stipulations set forth the agreed terms for the delivery of the Premises and the completion of improvements.",
        "The parties have initialed the Special Stipulations attached as Exhibit D, which address the use of the loading dock.",
        "Special Stipulations concerning the lease of the Building's rooftop for the installation of antennas are attached.",
        "The parties agree that the Special Stipulations shall prevail over the printed terms of this Lease in the event of conflict.",
        "The Special Stipulations regarding the parking allocation for the Premises are set forth on the attached page.",
        "The parties have agreed to the Special Stipulations concerning the abatement of rent during the initial fit-out period.",
        "Special Stipulations addressing the obligations of the parties with respect to hazardous materials are attached.",
        "The parties acknowledge the Special Stipulations regarding the maintenance of the Premises' fire suppression system.",
        "The Special Stipulations set forth the terms under which {T} may install a secondary entrance to the Premises.",
        "The parties have attached Special Stipulations concerning the allocation of the cost of the Building's common areas.",
        "Special Stipulations regarding the security requirements for the Premises are attached hereto and made a part hereof.",
        "The parties agree that the Special Stipulations regarding the use of the Premises as a medical office shall apply.",
        "Special Stipulations concerning the restoration of the Premises at the end of the Lease Term are attached.",
        "The parties have agreed to the Special Stipulations that permit {T} to install a helipad on the roof of the Building.",
        "Special Stipulations addressing the obligations of {L} to repair the Building's façade are attached.",
        "The parties acknowledge the Special Stipulations regarding the installation of an internal staircase in the Premises.",
        "Special Stipulations regarding the use of the Premises for the storage of sensitive materials are attached.",
        "The parties have agreed to the Special Stipulations that address the Building's compliance with sustainability codes.",
        "Special Stipulations concerning the terms for the early termination of this Lease are attached.",
        "The parties acknowledge the Special Stipulations regarding the payment of the Tenant Improvement allowance.",
        "Special Stipulations addressing the rights of {T} to use the Building's roof for telecommunication equipment are attached.",
    ],
    "compalsory_reconstraction": [
        "In the event that the Building is damaged and the applicable insurance proceeds are insufficient, {L} may elect to demolish the Building and terminate this Lease.",
        "The parties agree that if the Premises are destroyed, {L} shall have the right to reconstruct the same within a reasonable period of time.",
        "In the event of the substantial destruction of the Building, {L} shall have the right to terminate this Lease.",
        "The parties agree that if the Premises are damaged by fire or other casualty, {L} shall diligently repair and restore the same.",
        "In the event that the Building is condemned, this Lease shall terminate as of the date the condemnation is effective.",
        "The parties agree that {L} shall be obligated to reconstruct the Premises if the damage is not occasioned by {T}'s negligence.",
        "In the event of the partial destruction of the Premises, {L} shall have the right to terminate this Lease.",
        "The parties agree that if the Premises are rendered untenantable, {L} shall have a reasonable time to reconstruct the same.",
        "In the event that the Building is damaged during the last year of the Lease Term, {L} may terminate this Lease.",
        "The parties agree that {L} shall not be obligated to rebuild the Premises if the damage is the result of {T}'s breach.",
        "In the event of the demolition of the Building by the applicable authorities, this Lease shall terminate.",
        "The parties agree that {L} shall have the right to elect whether to reconstruct the Premises in the event of damage.",
        "In the event that the Premises are substantially destroyed, {T} shall have the right to terminate this Lease.",
        "The parties agree that {L} shall reconstruct the Premises in a manner consistent with the then-existing Building standards.",
        "In the event that the cost of reconstruction exceeds the insurance proceeds, {L} shall have the right to terminate this Lease.",
        "The parties agree that the obligations of {L} to reconstruct the Premises shall be subject to the applicable building permits.",
        "In the event of the total destruction of the Premises, this Lease shall terminate automatically.",
        "The parties agree that {L} shall have the right to terminate this Lease if the reconstruction would require {years} years or more.",
        "In the event that the Premises are damaged by a casualty, {T} shall cooperate with {L} in the reconstruction process.",
        "The parties agree that {L} shall be responsible for the reconstruction of the Premises following a casualty.",
        "In the event that the Building is condemned in part, the Base Rent shall be abated proportionately.",
        "The parties agree that {L} shall have the right to demolish the Building and terminate this Lease upon {days} days' notice.",
        "In the event of the destruction of the Building, the parties' obligations under this Lease shall terminate.",
        "The parties agree that {L} shall reconstruct the Premises using materials of a quality comparable to the original.",
        "In the event that the Premises are destroyed during the Lease Term, this Lease shall terminate as of the date of destruction.",
    ],
    "warrantees_of_the_owner": [
        "{L} warrants that it has good and marketable title to the Building and the Premises.",
        "{L} represents that it has the full right and authority to enter into this Lease.",
        "{L} warrants that there are no existing encumbrances on the Premises other than those disclosed herein.",
        "{L} represents that the Premises comply with all applicable building codes and regulations.",
        "{L} warrants that the Premises are free of all liens and that none shall attach during the Lease Term.",
        "{L} represents that it has obtained all necessary approvals to enter into this Lease.",
        "{L} warrants that the Building is not subject to any pending condemnation proceeding.",
        "{L} represents that the systems serving the Premises are in good working order as of the Commencement Date.",
        "{L} warrants that it is not a party to any litigation that would materially affect its ability to perform this Lease.",
        "{L} represents that the Premises are not subject to any environmental contamination of which {L} is aware.",
        "{L} warrants that the leases of the other tenants in the Building do not grant rights inconsistent with this Lease.",
        "{L} represents that all real property taxes with respect to the Building have been paid to date.",
        "{L} warrants that there are no unpaid assessments or charges against the Premises.",
        "{L} represents that the Building's certificate of occupancy is valid and current.",
        "{L} warrants that it has the power and authority to perform its obligations under this Lease.",
        "{L} represents that the Premises have not been used for any purpose that would violate applicable law.",
        "{L} warrants that it will keep the Premises free of any liens during the Lease Term.",
        "{L} represents that the information set forth in the Basic Lease Information is true and correct.",
        "{L} warrants that no broker is entitled to a commission from {T} in connection with this Lease.",
        "{L} represents that it has not received notice of any violation with respect to the Building.",
        "{L} warrants that the Premises are structurally sound as of the Commencement Date.",
        "{L} represents that it will provide {T} with quiet enjoyment of the Premises during the Lease Term.",
        "{L} warrants that the Building's common areas are in a safe and code-compliant condition.",
        "{L} represents that it is not subject to any order affecting its right to lease the Premises.",
        "{L} warrants that the Premises are not located in a flood zone as designated by applicable authorities.",
    ],
    "break_option": [
        "{T} shall have the right to terminate this Lease upon {days} days' written notice and the payment of a termination fee.",
        "The parties agree that {T} may terminate this Lease early upon the payment of the applicable termination fee set forth in {num}.",
        "{T} shall have the option to terminate this Lease at the end of the third Lease Year upon written notice.",
        "The parties agree that {T} may terminate this Lease at any time after the initial {years} years of the term.",
        "{T} shall have the right to terminate this Lease upon the payment of a termination fee equal to {amt}.",
        "The parties agree that {T} may elect to terminate this Lease early upon the delivery of the required notice.",
        "{T} shall have the option to terminate this Lease at the end of the fifth Lease Year upon the terms set forth herein.",
        "The parties agree that {T} shall have the right to terminate this Lease upon {days} days' notice if the Premises are not delivered.",
        "{T} shall have the right to terminate this Lease upon the payment of a termination fee representing {years} months of Base Rent.",
        "The parties agree that {T} may terminate this Lease in connection with a relocation of its headquarters.",
    ],
    "change_of_control": [
        "The parties agree that a change of control of {T} shall not constitute an assignment for the purposes of this Lease.",
        "In the event of a change of control of {T}, {L} shall have the right to terminate this Lease.",
        "The parties agree that a change of control of {L} shall not terminate this Lease or affect {T}'s rights.",
        "A change of control of {T} shall require the consent of {L}, which consent shall not be unreasonably withheld.",
        "The parties agree that {T} shall provide {L} with written notice of any change of control within {days} days.",
        "In the event of a change of control of {T}, this Lease shall continue in full force and effect.",
        "The parties agree that a merger of {T} with another entity shall be deemed a change of control.",
        "A change of control of {L} shall not release {L} from its obligations under this Lease.",
        "The parties agree that {T} shall not effect a change of control without the prior written consent of {L}.",
        "In the event of a change of control of {T}, the transferee shall assume the obligations of {T} under this Lease.",
    ],
    "damage": [
        "If the Premises are damaged by fire or other casualty, {L} shall repair the same within a reasonable period of time.",
        "The parties agree that {T} shall promptly notify {L} of any damage to the Premises.",
        "In the event of damage to the Premises, the Base Rent shall be abated proportionately during the period of repair.",
        "The parties agree that {T} shall be responsible for the repair of any damage to the Premises caused by {T}.",
        "If the Premises are damaged to the extent that they are rendered untenantable, this Lease may be terminated.",
        "The parties agree that {L} shall maintain insurance covering damage to the Building caused by fire or other casualty.",
        "In the event of damage to the Premises, {T} shall continue to pay the Base Rent until the damage is repaired.",
        "The parties agree that {T} shall have the right to terminate this Lease if the damage is not repaired within {days} days.",
        "If the Premises are substantially damaged, {L} shall have the right to terminate this Lease.",
        "The parties agree that the obligations of {L} to repair damage shall be subject to the receipt of insurance proceeds.",
    ],
    "guarantee_transferable": [
        "The guarantee of {T}'s obligations under this Lease shall be transferable and binding on the successors of the guarantor.",
        "The parties agree that the obligations of the guarantor shall survive any transfer of this Lease.",
        "The guarantor acknowledges that its obligations under this Lease are transferable to any assignee of {T}.",
        "The parties agree that the guarantee shall be binding upon the estate and successors of the guarantor.",
        "The guarantor agrees that the guarantee shall continue in full force and effect notwithstanding any assignment of this Lease.",
        "The parties agree that the guarantee of {T}'s obligations shall inure to the benefit of {L}'s successors and assigns.",
        "The guarantor waives any right to be released from its obligations upon a transfer of this Lease.",
        "The parties agree that the guarantee shall be assignable by {L} in connection with the transfer of the Building.",
        "The guarantor confirms that its obligations are not released by any change in the identity of {T}.",
        "The parties agree that the guarantee shall remain in effect for the entire Lease Term and any extension thereof.",
    ],
    "holdover": [
        "If {T} holds over after the expiration of the Lease Term, the tenancy shall be on a month-to-month basis at a rent equal to {amt}.",
        "The parties agree that any holdover by {T} shall constitute a tenancy at sufferance.",
        "If {T} remains in possession after the Expiration Date, {T} shall pay rent at a rate equal to one hundred fifty percent of the Base Rent.",
        "The parties agree that {T} shall be liable for any damages caused to {L} by its holdover.",
        "If {T} fails to vacate the Premises upon the expiration of the Lease Term, {T} shall pay double the monthly Base Rent.",
        "The parties agree that a holdover by {T} shall not extend the Lease Term or create a new lease.",
        "If {T} holds over for a period exceeding {days} days, {L} may elect to treat the holdover as a renewal of this Lease.",
        "The parties agree that {T} shall vacate the Premises upon the Expiration Date and deliver the same to {L} in good condition.",
        "If {T} holds over without the consent of {L}, {L} shall be entitled to recover the fair market rent for the Premises.",
        "The parties agree that any holdover tenancy shall be subject to all of the terms and conditions of this Lease.",
    ],
    "landlord_repairs": [
        "{L} shall repair and maintain the structural components of the Building, including the roof and exterior walls.",
        "The parties agree that {L} shall be responsible for the maintenance of the Building's common areas.",
        "{L} shall keep the elevators, stairs, and public corridors of the Building in good working order.",
        "The parties agree that {L} shall repair the Building's mechanical and electrical systems serving the Premises.",
        "{L} shall maintain the parking areas serving the Premises in good condition and adequate lighting.",
        "The parties agree that {L} shall be responsible for all repairs to the roof of the Building.",
        "{L} shall promptly make any repairs required to comply with applicable laws and codes.",
        "The parties agree that {L} shall maintain the Building's fire and life safety systems in working order.",
        "{L} shall repair any damage to the Premises occasioned by the negligence of {L} or its employees.",
        "The parties agree that {L} shall be responsible for the repair and replacement of the Building's HVAC equipment.",
    ],
    "reinstatement_clause": [
        "Upon the cure of any default by {T}, this Lease shall be reinstated as if no default had occurred.",
        "The parties agree that {T} shall have the right to reinstate this Lease upon the cure of the applicable default.",
        "If {T} cures any default within the applicable cure period, this Lease shall be reinstated.",
        "The parties agree that the reinstatement of this Lease shall be subject to the payment of all amounts in arrears.",
        "Upon the reinstatement of this Lease, the terms and conditions hereof shall remain in full force and effect.",
        "The parties agree that {T} shall have the right to reinstate this Lease if it cures the default within {days} days.",
        "If this Lease is terminated as a result of a default, {T} shall have the right to reinstate the same upon cure.",
        "The parties agree that the reinstatement option shall be exercised by {T} within a reasonable period of time.",
        "Upon reinstatement of this Lease, the obligations of the parties shall be revived as of the date of reinstatement.",
        "The parties agree that any reinstatement of this Lease shall not constitute a new lease.",
    ],
    "right_of_first_refusal_to_lease": [
        "{T} shall have a right of first refusal to lease the expansion premises when they become available.",
        "The parties agree that {L} shall offer the expansion space to {T} before offering it to any third party.",
        "{T} shall have the right of first refusal with respect to the space on the same floor of the Building.",
        "The parties agree that {L} shall give {T} written notice of any proposed lease of the expansion premises.",
        "{T} shall have the right to lease the expansion premises upon the same terms offered to a third party.",
        "The parties agree that the right of first refusal shall expire if {T} fails to respond within {days} days.",
        "{T} shall have the first right to lease the space on the mezzanine level of the Building.",
        "The parties agree that {T} shall have a right of first refusal to lease the space across the hall.",
        "{T} shall have the right to lease the expansion premises at the then-prevailing market rent.",
        "The parties agree that the right of first refusal shall be exercisable by {T} upon written notice to {L}.",
    ],
    "right_of_first_refusal_to_purchase": [
        "{T} shall have a right of first refusal to purchase the Building if {L} elects to sell the same.",
        "The parties agree that {L} shall give {T} written notice of any bona fide offer for the Building.",
        "{T} shall have the right to purchase the Building upon the terms set forth in {num}.",
        "The parties agree that {T} shall have the right of first refusal to purchase the Premises.",
        "{T} shall have the right to match any bona fide offer for the Building made by a third party.",
        "The parties agree that the right of first refusal to purchase shall be exercisable within {days} days of notice.",
        "{T} shall have the option to purchase the Building at a price equal to the applicable market value.",
        "The parties agree that {T} shall have the first right to purchase the Premises upon the terms agreed.",
        "{T} shall have the right of first refusal to acquire the Building from {L} upon the terms of any third-party offer.",
        "The parties agree that the purchase option shall terminate upon the expiration of the Lease Term.",
    ],
    "services_charges": [
        "{T} shall pay to {L} its proportionate share of the costs of operating and maintaining the Building.",
        "The parties agree that the Operating Expenses shall be allocated to {T} in accordance with {num}.",
        "{T} shall pay an additional rent amount equal to its proportionate share of the Common Area Maintenance costs.",
        "The parties agree that {T} shall pay its proportionate share of the costs of the Building's security services.",
        "{T} shall pay to {L} its share of the cost of the Building's janitorial services.",
        "The parties agree that the operating expense escalation shall be calculated in accordance with the terms set forth herein.",
        "{T} shall pay its proportionate share of the cost of the Building's utilities to the extent not separately metered.",
        "The parties agree that {T} shall pay its share of the real property taxes assessed against the Building.",
        "{T} shall pay to {L} an amount equal to its proportionate share of the cost of the Building's insurance.",
        "The parties agree that the additional rent for operating expenses shall be paid within {days} days of the statement.",
    ],
    "sublease_permitted": [
        "{T} may sublease the Premises with the prior written consent of {L}, which consent shall not be unreasonably withheld.",
        "The parties agree that {T} may sublet a portion of the Premises upon the written consent of {L}.",
        "{T} shall be permitted to sublease the Premises to a third party subject to the approval of {L}.",
        "The parties agree that {T} may sublet the Premises provided that the subtenant uses the same for a permitted use.",
        "{T} shall have the right to sublease the Premises with the prior written consent of {L}.",
        "The parties agree that any sublease of the Premises shall be subject to the terms and conditions of this Lease.",
        "{T} may sublet the Premises upon written notice to {L}, with consent not to be unreasonably withheld.",
        "The parties agree that {T} shall remain liable to {L} for the performance of all obligations under any sublease.",
        "{T} shall be permitted to sublet the Premises to an affiliate upon written notice to {L}.",
        "The parties agree that {T} may sublet the Premises for the remainder of the Lease Term upon {L}'s consent.",
    ],
    "none": [
        "The Premises are leased for a term commencing on the Commencement Date and ending on the Expiration Date.",
        "The Base Rent payable under this Lease is the amount set forth in the Basic Lease Information.",
        "The Building is located at the address specified in the first paragraph of this Lease.",
        "Notices to the parties shall be delivered to the addresses set forth in the Basic Lease Information.",
        "The Security Deposit required under this Lease is the amount identified in the first paragraph hereof.",
        "This Lease shall be effective upon the execution and delivery hereof by the parties.",
        "The parties have agreed upon the number of parking spaces allocated to the Premises.",
        "The Lease Term shall be the period specified in the Basic Lease Information.",
        "The Premises are located on the floor and in the location described in the floor plan.",
        "The parties have executed this Lease in the manner required by law.",
        "The Rent Commencement Date is the date on which the Base Rent first becomes payable.",
        "The Premises are delivered in the condition described in the work letter attached as Exhibit B.",
        "The parties have initialed each page of this Lease as evidence of their agreement.",
        "The Building's address is set forth in the first paragraph of this Lease.",
        "The parties have agreed that the Base Rent shall be payable in equal monthly installments.",
        "The exhibits and schedules listed in this Lease are attached hereto.",
        "The parties acknowledge that the Premises are located in the Building described above.",
        "The Lease has been executed by the parties in the presence of the witnesses identified below.",
        "The parties have agreed upon the permitted use of the Premises.",
        "The effective date of this Lease is set forth on the first page hereof.",
        "The parties have acknowledged the receipt of the Security Deposit from the Tenant.",
        "The Commencement Date and the Expiration Date are set forth in the Basic Lease Information.",
        "The parties agree that the Lease Term shall be measured from the Commencement Date.",
        "The Premises are a portion of the Building identified in the floor plan.",
        "The parties have agreed that the Base Rent shall escalate annually as set forth in this Lease.",
        "The parties acknowledge that the Premises are subject to the Building's rules and regulations.",
        "The Lease has been reviewed by the attorneys for the respective parties.",
        "The parties agree that the effective date is the date of the last signature hereto.",
        "The parties have agreed upon the square footage of the Premises.",
        "The Lease and its exhibits constitute the complete agreement of the parties.",
        "The parties acknowledge the receipt of the keys to the Premises.",
        "The Building provides the services described in the work letter.",
        "The parties have agreed that the Base Rent is payable without deduction or offset.",
        "The parties have identified the brokers involved in this transaction.",
        "The Premises shall be used for the purpose set forth in the Basic Lease Information.",
        "The parties acknowledge that the Building is subject to the covenants described in the title report.",
        "The Lease Term shall commence upon the delivery of possession of the Premises.",
        "The parties have agreed to the schedule for the payment of estimated operating expenses.",
        "The parties acknowledge that the Premises are delivered in their current condition.",
        "The parties have agreed that the Lease shall be recorded in the applicable registry.",
    ],
}


def _unique(rows: list[dict], label: str) -> list[dict]:
    """Ensure source ids and (for dedup sanity) raw text are unique."""
    seen: set[str] = set()
    for r in rows:
        if r["source"] in seen:
            raise RuntimeError(f"{label}: duplicate source {r['source']}")
        seen.add(r["source"])
    return rows


def generate_multilabel(seed: int, per_category: int = DEONTIC_PER_CATEGORY) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i, cat in enumerate(DEONTIC_ORDER):
        tpls = DEONTIC_TEMPLATES[cat]
        for j in range(per_category):
            tpl = tpls[j % len(tpls)]
            variant = (j // len(tpls)) + i * 131
            text = _render(tpl, variant)
            party = "tenant" if rng.random() < 0.5 else "landlord"
            label = [0] * len(DEONTIC_ORDER)
            label[DEONTIC_LABEL[cat]] = 1
            rows.append(
                {
                    "source": f"d-{cat}-{j + 1:03d}",
                    "sentence_idx": 1,
                    "party": party,
                    "text": text,
                    "label": label,
                    "deontic_types": [cat],
                }
            )
    return _unique(rows, "deontic_multilabel")


def generate_redflag(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for cls, count in REDFLAG_COUNTS.items():
        tpls = REDFLAG_TEMPLATES[cls]
        for j in range(count):
            tpl = tpls[j % len(tpls)]
            variant = (j // len(tpls)) + rng.randrange(0, 100)
            rows.append(
                {
                    "source": f"rf-{cls}-{j + 1:03d}",
                    "text": "",
                    "raw_text": _render(tpl, variant),
                    "type": cls,
                    "start": 0,
                    "end": 0,
                }
            )
    return _unique(rows, "redflag_paragraph")


def generate_span(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for cls in REDFLAG_COUNTS:
        if cls == "none":
            continue
        tpls = REDFLAG_TEMPLATES[cls]
        per = 8 if cls in SPAN_RARE else 4
        for j in range(per):
            tpl = tpls[j % len(tpls)]
            variant = (j // len(tpls)) + rng.randrange(0, 100)
            text = _render(tpl, variant)
            raw = text
            if j % 2 == 0:
                num = CLAUSE_NUMS[j % len(CLAUSE_NUMS)]
                raw = f"{num} {text}"
            rows.append(
                {
                    "source": f"sp-{cls}-{j + 1:02d}",
                    "part": f"s1p{j + 1:02d}",
                    "text": text,
                    "raw_text": raw,
                    "type": cls,
                    "start": 0,
                    "end": 0,
                }
            )
    return _unique(rows, "deontic_span")


def write_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


def _real_vocab(name: str) -> set[str]:
    vocab: set[str] = set()
    root = Path(__file__).resolve().parent.parent / "data" / "splits"
    for split in ("train", "val", "test"):
        p = root / f"{name}.{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            vocab.add(json.loads(line)["type"])
    return vocab


def validate(rows: list[dict], label: str, vocab: set[str] | None = None) -> int:
    """Reuse the ingest_synthetic validation rules; returns number of rows."""
    from ingest_synthetic import validate_multilabel, validate_redflag, validate_span

    if label == "deontic_multilabel":
        validate_multilabel(rows)
    elif label == "redflag_paragraph":
        rows = validate_redflag(rows, vocab or set())
    elif label == "deontic_span":
        rows = validate_span(rows, vocab or set())
    print(f"  [ok] {label}: {len(rows)} rows validated")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_generated", help="output dir (default data/synthetic_generated)")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / args.out
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    print(f"[build_synthetic] seed={args.seed} out={out}")

    dm = generate_multilabel(args.seed)
    rf = generate_redflag(args.seed)
    sp = generate_span(args.seed)

    n_dm = write_jsonl(dm, out / "deontic_multilabel.jsonl")
    n_rf = write_jsonl(rf, out / "redflag_paragraph.jsonl")
    n_sp = write_jsonl(sp, out / "deontic_span.jsonl")
    print(f"[build_synthetic] wrote {n_dm} deontic_multilabel, {n_rf} redflag_paragraph, {n_sp} deontic_span")

    print("[build_synthetic] validating ...")
    validate(dm, "deontic_multilabel")
    validate(rf, "redflag_paragraph", vocab=_real_vocab("redflag_paragraph"))
    validate(sp, "deontic_span", vocab=_real_vocab("deontic_span"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
