"""Invariants for the statute grounding corpus.

The shipped hand-built index shared one chunk id per document, kept title/TOC/
background-note pages, and contained mojibake OCR pages - so citations were
non-unique and retrieval degenerated to the title page. These tests pin the
rebuild contract: unique section-keyed ids and no junk chunks.
"""
from __future__ import annotations

from pathlib import Path

from legalrag.retrieve.statutes import chunkAct, cleanChunks, isJunk

FIXTURE = """# Model Tenancy Act 2021 (India)

Page 1 of 3
BACKGROUND NOTE ON MODEL TENANCY ACT (MTA)
As per Census 2011 around 110 lakh houses were lying vacant in urban areas.

Page 2 of 3
MTA will enable unlocking of vacant premises for rental purposes.

CHAPTER I
PRELIMINARY
SECTIONS
1. Short title, extent and commencement.
2. Definitions.
3. Act not to apply to certain premises.
CHAPTER II
PROVISIONS REGARDING RENT
4. Tenancy agreement.
5. Period of tenancy.

Page 4 of 27
THE MODEL TENANCY ACT, 2020
An Act to establish Rent Authority to regulate renting of premises and to
protect the interests of landlords and tenants.
BE it enacted by the Legislature as follows:--

CHAPTER I
PRELIMINARY
1. (1) This Act may be called the Tenancy Act, 2020.
(2) It shall come into force on such date as the State Government may
appoint, and different dates may be appointed for different provisions.
2. (1) In this Act, unless the context otherwise requires, "tenant" means
a person by whom rent is payable for the premises.
3. This Act shall not apply to premises owned or occupied by the
Government, a local authority, or any premises used for non-residential
purposes exceeding a monthly rent of fifty thousand rupees.

CHAPTER II
PROVISIONS REGARDING RENT
4. (1) Every tenancy agreement shall be in writing and registered.
(2) The tenant shall be entitled to a rent receipt for every payment
made, and failure to provide a receipt shall attract a penalty.
5. (1) The period of tenancy shall be as agreed between the parties.
(2) The landlord shall not evict a tenant without a valid ground as
specified in this Act during the fixed period.

6. (1) The rent payable shall not be increased more than once in twelve
months, and the increase shall not exceed a prescribed percentage.

First Schedule
Second Schedule
"""


def test_chunk_ids_unique_and_section_keyed(tmp_path: Path) -> None:
    src = tmp_path / "mta_2021.md"
    src.write_text(FIXTURE, encoding="utf-8")

    chunks = cleanChunks(chunkAct(src, "mta_2021"))

    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    assert ids == [f"mta_2021#s.{n}" for n in (1, 2, 3, 4, 5, 6)], ids


def test_no_junk_chunks(tmp_path: Path) -> None:
    src = tmp_path / "mta_2021.md"
    src.write_text(FIXTURE, encoding="utf-8")

    chunks = cleanChunks(chunkAct(src, "mta_2021"))

    for chunk in chunks:
        head = chunk["text"]
        assert "ARRANGEMENT OF SECTIONS" not in head
        assert "BACKGROUND NOTE" not in head
        assert "Page 1 of" not in head and "Page 4 of" not in head
        assert not head.startswith("# Model")
        assert not head.startswith("THE MODEL TENANCY ACT")


def test_short_toc_entries_dropped(tmp_path: Path) -> None:
    src = tmp_path / "mta_2021.md"
    src.write_text(FIXTURE, encoding="utf-8")

    raw = chunkAct(src, "mta_2021")
    assert not [c for c in raw if "Short title, extent" in c["text"]]


def test_wrapped_toc_entries_dropped(tmp_path: Path) -> None:
    src = tmp_path / "delhi_rent_control_1958.md"
    src.write_text(
        "# Delhi Rent Control Act 1958\n"
        "ARRANGEMENT OF SECTIONS\n"
        "CHAPTER I\nPRELIMINARY\nSECTIONS\n"
        "1. Short title, extent and commencement.\n"
        "14C. Right to recover immediate possession of premises to accrue to Central Government and Delhi \n"
        "Administration employees.\n"
        "THE FIRST SCHEDULE.\n"
        "ACT NO. 59 OF 1958\n"
        "BE it enacted by Parliament in the Ninth Year of the Republic of India as follows:--\n"
        "CHAPTER I\nPRELIMINARY\n"
        "1. (1) This Act may be called the Delhi Rent Control Act, 1958.\n"
        "(2) It extends to the whole of the Union territory of Delhi.\n"
        "14C. (1) Where a person is a member of the armed forces and the premises\n"
        "were let out to him, he may recover immediate possession of the premises.\n",
        encoding="utf-8",
    )

    chunks = cleanChunks(chunkAct(src, "delhi_rent_control_1958"))

    ids = [c["id"] for c in chunks]
    assert ids == ["delhi_rent_control_1958#s.1", "delhi_rent_control_1958#s.14C"], ids
    assert not [c for c in chunks if "Administration employees." in c["text"]]


def test_mojibake_pages_dropped(tmp_path: Path) -> None:
    src = tmp_path / "mta_2021.md"
    src.write_text(
        FIXTURE
        + "\n\nपçठृ 29 का 25 सदèय, आिद का लोकसेवक होना। 42. इस अिधिनयम के अधीन िनयु क्त िकराया\n"
        + "पçठृ 29 का 26 (छ) धारा 35 की उपधारा (5) के अधीन िकराया अिधकरण के समक्ष अपील\n",
        encoding="utf-8",
    )

    chunks = cleanChunks(chunkAct(src, "mta_2021"))

    assert not [c for c in chunks if "\u0900" <= c["text"][:1] <= "\u097f"]


def test_is_junk() -> None:
    assert isJunk("Page 2 of 3\nMTA will enable unlocking of vacant premises.")
    assert isJunk("# Model Tenancy Act 2021 (India)")
    assert isJunk("1. Short title, extent and commencement.")
    assert isJunk("पçठृ 29 का 25 सदèय")
    assert not isJunk(
        "2. (1) In this Act, unless the context otherwise requires, 'tenant' means "
        "a person by whom rent is payable for the premises and includes any "
        "person continuing in possession after the termination of tenancy."
    )


def test_percent_like_lines_not_treated_as_section_headers(tmp_path: Path) -> None:
    src = tmp_path / "mta_2021.md"
    src.write_text(
        "The share of urban population has increased to 31.16% in 2011 as "
        "compared to 27.82% in 1991 and further urban population projected "
        "to be greater than fifty per cent by 2050, driven largely by "
        "migration from rural areas into cities for education, employment, "
        "healthcare and a better quality of life.\n",
        encoding="utf-8",
    )

    chunks = cleanChunks(chunkAct(src, "mta_2021"))
    assert chunks == []