"""Canonical accounting concept specifications for EDGAR pulls.

This module centralizes the quarterly accounting concept map so the raw pull,
cleaning logic, and diagnostics use the same canonical definitions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AccountingConceptSpec:
    canonical_name: str
    candidate_tags: tuple[str, ...]
    preferred_unit: str
    conflict_resolution_rule: str


CONCEPT_SPECS: dict[str, AccountingConceptSpec] = {
    "revenue": AccountingConceptSpec(
        canonical_name="revenue",
        candidate_tags=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
            "Revenue",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then the highest-priority tag in the candidate list, "
            "then the latest filing_date, then the preferred source."
        ),
    ),
    "gross_profit": AccountingConceptSpec(
        canonical_name="gross_profit",
        candidate_tags=("GrossProfit",),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then GrossProfit, then the latest filing_date."
        ),
    ),
    "cogs": AccountingConceptSpec(
        canonical_name="cogs",
        candidate_tags=(
            "CostOfGoodsSold",
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfSales",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then the most direct COGS tag, then the latest filing_date."
        ),
    ),
    "operating_income": AccountingConceptSpec(
        canonical_name="operating_income",
        candidate_tags=("OperatingIncomeLoss",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then OperatingIncomeLoss, then the latest filing_date.",
    ),
    "ebit": AccountingConceptSpec(
        canonical_name="ebit",
        candidate_tags=(
            "OperatingIncomeLoss",
            "IncomeBeforeTaxExpenseBenefitAndInterestExpense",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeBeforeTaxExpenseBenefit",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then the listed EBIT-like tags in priority order, then the latest filing_date."
        ),
    ),
    "ebitda": AccountingConceptSpec(
        canonical_name="ebitda",
        candidate_tags=(
            "EarningsBeforeInterestTaxesDepreciationAndAmortization",
            "AdjustedEBITDA",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then the listed EBITDA tags, then the latest filing_date.",
    ),
    "net_income": AccountingConceptSpec(
        canonical_name="net_income",
        candidate_tags=("NetIncomeLoss", "ProfitLoss"),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then NetIncomeLoss, then the latest filing_date.",
    ),
    "sga": AccountingConceptSpec(
        canonical_name="sga",
        candidate_tags=(
            "SellingGeneralAndAdministrativeExpense",
            "SellingAndMarketingExpense",
            "GeneralAndAdministrativeExpense",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then full SG&A, then narrower SG&A-related tags, then the latest filing_date."
        ),
    ),
    "r_and_d": AccountingConceptSpec(
        canonical_name="r_and_d",
        candidate_tags=("ResearchAndDevelopmentExpense",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then ResearchAndDevelopmentExpense, then the latest filing_date.",
    ),
    "interest_expense": AccountingConceptSpec(
        canonical_name="interest_expense",
        candidate_tags=(
            "InterestExpenseAndDebtExpense",
            "InterestExpense",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then the broader interest-and-debt-expense tag, then the latest filing_date."
        ),
    ),
    "income_tax_expense": AccountingConceptSpec(
        canonical_name="income_tax_expense",
        candidate_tags=(
            "IncomeTaxExpenseBenefit",
            "IncomeTaxes",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then IncomeTaxExpenseBenefit, then the latest filing_date.",
    ),
    "total_assets": AccountingConceptSpec(
        canonical_name="total_assets",
        candidate_tags=("Assets",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then Assets, then the latest filing_date.",
    ),
    "total_liabilities": AccountingConceptSpec(
        canonical_name="total_liabilities",
        candidate_tags=("Liabilities",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then Liabilities, then the latest filing_date.",
    ),
    "current_assets": AccountingConceptSpec(
        canonical_name="current_assets",
        candidate_tags=("AssetsCurrent",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then AssetsCurrent, then the latest filing_date.",
    ),
    "current_liabilities": AccountingConceptSpec(
        canonical_name="current_liabilities",
        candidate_tags=("LiabilitiesCurrent",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then LiabilitiesCurrent, then the latest filing_date.",
    ),
    "cash_and_cash_equivalents": AccountingConceptSpec(
        canonical_name="cash_and_cash_equivalents",
        candidate_tags=(
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer unrestricted cash if present, otherwise the broader cash-and-restricted-cash tag, then the latest filing_date."
        ),
    ),
    "accounts_receivable": AccountingConceptSpec(
        canonical_name="accounts_receivable",
        candidate_tags=(
            "AccountsReceivableNetCurrent",
            "ReceivablesNetCurrent",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then AccountsReceivableNetCurrent, then the latest filing_date.",
    ),
    "inventory": AccountingConceptSpec(
        canonical_name="inventory",
        candidate_tags=("InventoryNet", "InventoriesNetOfReserves"),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then InventoryNet, then the latest filing_date.",
    ),
    "ppe_net": AccountingConceptSpec(
        canonical_name="ppe_net",
        candidate_tags=(
            "PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then PP&E net tags in priority order, then the latest filing_date.",
    ),
    "goodwill": AccountingConceptSpec(
        canonical_name="goodwill",
        candidate_tags=("Goodwill",),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then Goodwill, then the latest filing_date.",
    ),
    "intangible_assets": AccountingConceptSpec(
        canonical_name="intangible_assets",
        candidate_tags=(
            "FiniteLivedIntangibleAssetsNet",
            "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
            "IntangibleAssetsNetExcludingGoodwill",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then net non-goodwill intangibles, then the latest filing_date."
        ),
    ),
    "short_term_debt": AccountingConceptSpec(
        canonical_name="short_term_debt",
        candidate_tags=(
            "ShortTermBorrowings",
            "LongTermDebtCurrent",
            "ShortTermDebt",
            "CurrentPortionOfLongTermDebt",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then explicit short-term borrowings, then current debt portions, then the latest filing_date."
        ),
    ),
    "long_term_debt": AccountingConceptSpec(
        canonical_name="long_term_debt",
        candidate_tags=(
            "LongTermDebtAndCapitalLeaseObligations",
            "LongTermDebtNoncurrent",
            "LongTermDebt",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then the broadest long-term-debt tag, then the latest filing_date."
        ),
    ),
    "shareholders_equity": AccountingConceptSpec(
        canonical_name="shareholders_equity",
        candidate_tags=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then StockholdersEquity, then the latest filing_date.",
    ),
    "shares_outstanding": AccountingConceptSpec(
        canonical_name="shares_outstanding",
        candidate_tags=(
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
        ),
        preferred_unit="shares",
        conflict_resolution_rule=(
            "Prefer share-count units, then CommonStockSharesOutstanding, then the latest filing_date."
        ),
    ),
    "operating_cash_flow": AccountingConceptSpec(
        canonical_name="operating_cash_flow",
        candidate_tags=(
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then total operating cash flow, then continuing-operations OCF, then the latest filing_date."
        ),
    ),
    "capex": AccountingConceptSpec(
        canonical_name="capex",
        candidate_tags=(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpendituresIncurredButNotYetPaid",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer cash capex tags in USD, then the highest-priority capex tag, then the latest filing_date."
        ),
    ),
    "stock_based_compensation": AccountingConceptSpec(
        canonical_name="stock_based_compensation",
        candidate_tags=("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then ShareBasedCompensation, then the latest filing_date."
        ),
    ),
    "share_repurchases": AccountingConceptSpec(
        canonical_name="share_repurchases",
        candidate_tags=(
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfEquity",
        ),
        preferred_unit="USD",
        conflict_resolution_rule=(
            "Prefer USD facts, then common-stock repurchases, then broader equity repurchases, then the latest filing_date."
        ),
    ),
    "dividends_paid": AccountingConceptSpec(
        canonical_name="dividends_paid",
        candidate_tags=(
            "PaymentsOfDividends",
            "PaymentsOfOrdinaryDividends",
            "DividendsPaid",
        ),
        preferred_unit="USD",
        conflict_resolution_rule="Prefer USD facts, then dividend payment tags in priority order, then the latest filing_date.",
    ),
}


EXPECTED_CONCEPT_COLUMNS = list(CONCEPT_SPECS.keys())


UNLOCKED_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "gross_margin": ("revenue", "gross_profit"),
    "inventory_turnover": ("cogs", "inventory"),
    "interest_coverage": ("ebit", "interest_expense"),
    "capex_intensity": ("capex", "revenue"),
    "free_cash_flow": ("operating_cash_flow", "capex"),
    "free_cash_flow_margin": ("operating_cash_flow", "capex", "revenue"),
    "free_cash_flow_to_net_income": ("operating_cash_flow", "capex", "net_income"),
    "total_debt_to_assets": ("short_term_debt", "long_term_debt", "total_assets"),
    "leverage_change_qoq": ("short_term_debt", "long_term_debt", "total_assets"),
    "sga_to_revenue": ("sga", "revenue"),
    "r_and_d_to_revenue": ("r_and_d", "revenue"),
    "shareholder_payout_ratio": (
        "share_repurchases",
        "dividends_paid",
        "operating_cash_flow",
    ),
}


def concept_priority_lookup() -> dict[str, dict[str, int]]:
    return {
        canonical_name: {tag: rank for rank, tag in enumerate(spec.candidate_tags)}
        for canonical_name, spec in CONCEPT_SPECS.items()
    }


def source_priority(source_name: str | None) -> int:
    if source_name == "edgartools_companyfacts":
        return 0
    if source_name == "sec_companyfacts":
        return 1
    return 9


def export_concept_map(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(spec) for spec in CONCEPT_SPECS.values()]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path
