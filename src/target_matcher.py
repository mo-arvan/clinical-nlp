"""
Based on medspacy
TargetMatcher (medspacy/target_matcher/target_matcher.py)
MedspacyMatcher (medspacy/common/medspacy_matcher.py)
"""
from types import NoneType
from typing import Tuple, List

from medspacy.common.regex_matcher import RegexMatcher
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span

DICTIONARY = {"lol": "laughing out loud", "brb": "be right back"}
DICTIONARY.update({value: key for key, value in DICTIONARY.items()})


def tuple_overlaps(a: Tuple[int, int], b: Tuple[int, int]):
    """
    Calculates whether two tuples overlap. Assumes tuples are sorted to be like spans (start, end)

    Args:
        a: A tuple representing a span (start, end).
        b: A tuple representing a span (start, end).

    Returns:
        Whether the tuples overlap.
    """
    return a[0] <= b[0] < a[1] or a[0] < b[1] <= a[1]


def overlaps(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    """
    Checks whether two match Tuples out of spacy matchers overlap.

    Args:
        a: A match Tuple (match_id, start, end).
        b: A match Tuple (match_id, start, end).

    Returns:
        Whether the tuples overlap.
    """
    _, a_start, a_end = a
    _, b_start, b_end = b
    return tuple_overlaps((a_start, a_end), (b_start, b_end))


@Language.factory("target_matcher")
def create_acronym_component(nlp: Language, name: str):
    return TargetMatcher(nlp, name)


class TargetMatcher:
    def __init__(self, nlp: Language, name: str = "rule_matcher", phrase_matcher_attr: str = "LOWER"):
        # Create the matcher and match on Token.lower if case-insensitive
        self.nlp = nlp.tokenizer  # preserve only the tokenizer for creating phrasematcher rules
        self.regex_matcher = RegexMatcher(nlp.vocab)
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr=phrase_matcher_attr)
        self.phrase_matcher_attr = phrase_matcher_attr
        self.rule_count = 0

        self.overlapping_rules = []

        self.rule_map = {}

    def prune_overlapping_matches(self,
                                  matches: List[Tuple[int, int, int]], strategy: str = "longest"
                                  ) -> List[Tuple[int, int, int]]:
        """
        Prunes overlapping matches from a list of spaCy match tuples (match_id, start, end).

        Args:
            matches: A list of match tuples of form (match_id, start, end).
            strategy: The pruning strategy to use. At this time, the only available option is "longest" and will keep the
                longest of any two overlapping spans. Other behavior will be added in a future update.

        Returns:
            The pruned list of matches.
        """
        if strategy != "longest":
            raise NotImplementedError(
                "No other filtering strategy has been implemented. Coming in a future update."
            )

        # Make a copy and sort
        unpruned = sorted(matches, key=lambda x: (x[1], x[2]))
        pruned = []
        num_matches = len(matches)
        if num_matches == 0:
            return matches
        curr_match = unpruned.pop(0)

        while True:
            if len(unpruned) == 0:
                pruned.append(curr_match)
                break
            next_match = unpruned.pop(0)

            # Check if they overlap
            if overlaps(curr_match, next_match):
                current_match_rule = self.rule_map[self.nlp.vocab.strings[curr_match[0]]]
                next_match_rule = self.rule_map[self.nlp.vocab.strings[next_match[0]]]
                # Choose the larger span
                both_contain_metadata = (current_match_rule.metadata is not None and
                                         current_match_rule.metadata is not NoneType and
                                         next_match_rule.metadata is not None and
                                         next_match_rule.metadata is not NoneType)
                same_group = False
                if both_contain_metadata:
                    same_group = current_match_rule.metadata.get("group") == next_match_rule.metadata.get("group")

                if not both_contain_metadata or not same_group:
                    overlapping_rules = (current_match_rule.category, next_match_rule.category)
                    self.add_overlapping_rule(overlapping_rules)
                    longer_span = max(curr_match, next_match, key=lambda x: (x[2] - x[1]))
                    pruned.append(longer_span)
                    if len(unpruned) == 0:
                        break
                    curr_match = unpruned.pop(0)
                else:
                    current_match_priority = current_match_rule.metadata.get("priority", 0)
                    next_match_priority = next_match_rule.metadata.get("priority", 0)
                    if current_match_priority > next_match_priority:
                        pruned.append(curr_match)
                    else:
                        pruned.append(next_match)

                    if len(unpruned) == 0:
                        break
                    curr_match = unpruned.pop(0)
            else:
                pruned.append(curr_match)
                curr_match = next_match

        # Recursive base point
        if len(pruned) == num_matches:
            return pruned
        # Recursive function call
        else:
            return self.prune_overlapping_matches(pruned)


    def __call__(self, doc: Doc) -> Doc:
        # Add the matched spans when doc is processed
        regex_matches = self.regex_matcher(doc)
        phrase_matches = self.phrase_matcher(doc)

        all_matches = regex_matches + phrase_matches

        matches = self.prune_overlapping_matches(all_matches)

        if len(self.overlapping_rules) > 0:
            print(f"Overlapping rules: {self.overlapping_rules}")

        spans = []
        for rule_id, start, end in matches:
            try:
                rule = self.rule_map[self.nlp.vocab.strings[rule_id]]
                span = Span(doc, start=start, end=end, label=rule.category)
                span._.target_rule = rule
                if rule.attributes is not None:
                    for attribute, value in rule.attributes.items():
                        try:
                            setattr(span._, attribute, value)
                        except AttributeError as e:
                            raise e
                spans.append(span)
            except KeyError:
                # This should never happen, but just in case
                print(f"Rule ID {rule_id} not found in rule_map", RuntimeWarning)

        for span in spans:
            try:
                doc.ents += (span,)
            except ValueError:
                # spaCy will raise a value error if the token in span are already part of an entity (i.e., as part
                # of an upstream component). In that case, let the existing span supersede this one.
                print(
                    f'The result ""{span}"" conflicts with a pre-existing entity in doc.ents. This result has been '
                    f"skipped.",
                    RuntimeWarning,
                )
        return doc


    def add(self, rule_set):
        for rule in rule_set:
            rule_id = f"{rule.category}_{self.rule_count}"
            try:
                if isinstance(rule.pattern, str):
                    self.regex_matcher.add(rule_id, [rule.pattern])
                else:
                    if self.phrase_matcher_attr.lower() == "lower":
                        # only lowercase when the phrase matcher is looking for lowercase matches.
                        text = rule.literal.lower()
                    else:
                        # otherwise, expect users to handle phrases as aligned with their non-default phrase matching scheme
                        # this prevents .lower() from blocking matches on attrs like ORTH or UPPER
                        text = rule.literal
                    doc = self.nlp(text)
                    self.phrase_matcher.add(
                        rule_id,
                        [doc],
                        on_match=rule.on_match,
                    )

                self.rule_map[rule_id] = rule
                self.rule_count += 1


            except Exception as e:
                print(f"Failed to add rule {rule}")
                print(e)

    def add_overlapping_rule(self, rule_tuple):
        rule_tuple = tuple(sorted(rule_tuple))
        if rule_tuple not in self.overlapping_rules:
            self.overlapping_rules.append(rule_tuple)



def init():
    pass