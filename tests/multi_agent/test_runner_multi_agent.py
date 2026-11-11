"""
Test Runner — Multi-Agent Tests (Human-in-the-Loop)

Loads golden_dataset_multi_agent.csv, runs the full workflow for each test case,
and generates a plain text report for human review.

These tests exercise the complete pipeline: entity extraction → normalization →
supervisor routing → specialist(s) → synthesis. The report captures which
specialists ran, what they found, and the final synthesized answer.

No automated scoring or keyword matching — output is reviewed manually.

Report saved to:
  tests/multi_agent/reports/report_{timestamp}.txt
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))          # enables 'from src.x import ...'
sys.path.insert(0, str(project_root / "src"))  # enables 'from workflow.x import ...'

from workflow.graph_builder import build_workflow
from workflow.state import create_initial_state

load_dotenv()


class MultiAgentTestRunner:
    """Runs multi-agent end-to-end tests and generates a human-reviewable report."""

    def __init__(self, dataset_name: str = None, golden_dataset_path: str = None):
        """
        Initialize test runner.

        Args:
            dataset_name: Name of the dataset file without .csv extension.
                          Looks in the same folder as this script.
            golden_dataset_path: Direct path to CSV file (overrides dataset_name).
        """
        self.tests_dir = Path(__file__).parent
        self.test_type = "multi_agent"

        if golden_dataset_path:
            self.golden_dataset_path = Path(golden_dataset_path)
        elif dataset_name:
            self.golden_dataset_path = self.tests_dir / f"{dataset_name}.csv"
        else:
            self.golden_dataset_path = self.tests_dir / "golden_dataset_multi_agent.csv"

        if not self.golden_dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.golden_dataset_path}"
            )

        self.dataset_name = dataset_name or self.golden_dataset_path.stem
        self.test_cases = []
        self.results = []

        # Lazy initialization
        self.workflow = None

    def _initialize_workflow(self):
        """Initialize workflow. The Neo4j connection is created by connections.py the first time it is used."""
        print("🔧 Initializing workflow...")
        try:
            self.workflow = build_workflow()
            print("✓ Initialization complete\n")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from CSV file."""
        print(f"📖 Loading test cases from {self.golden_dataset_path}...")

        test_cases = []
        with open(self.golden_dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                medications = [m.strip() for m in row.get('medications', '').split(',') if m.strip()]
                supplements = [s.strip() for s in row.get('supplements', '').split(',') if s.strip()]
                conditions = [c.strip() for c in row.get('conditions', '').split(',') if c.strip()]
                dietary_restrictions = [d.strip() for d in row.get('dietary_restrictions', '').split(',') if d.strip()]

                test_case = {
                    'test_id': row.get('test_id', ''),
                    'user_question': row.get('user_question', ''),
                    'patient_profile': {
                        'medications': medications,
                        'supplements': supplements,
                        'conditions': conditions,
                        'dietary_restrictions': dietary_restrictions,
                    },
                    'test_type': row.get('test_type', ''),
                    'description': row.get('description', ''),
                }
                test_cases.append(test_case)

        self.test_cases = test_cases
        print(f"✓ Loaded {len(test_cases)} test cases\n")
        return test_cases

    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case through the full workflow.

        Captures the complete state including:
          - Entity extraction and normalization results
          - Which specialists ran (safety, deficiency, recommendation)
          - Each specialist's structured results
          - Supervisor routing decisions (via evidence_chain)
          - Final synthesized answer
        """
        if self.workflow is None:
            self._initialize_workflow()

        test_id = test_case['test_id']
        question = test_case['user_question']
        profile = test_case['patient_profile']

        print(f"🧪 Test {test_id}: {question[:60]}...")

        try:
            initial_state = create_initial_state(question, profile)
            state = self.workflow.invoke(initial_state)

            actual_answer = state.get('final_answer', '')
            if not actual_answer:
                actual_answer = state.get('error_message', 'No output generated')

            # Collect all specialist outputs
            safety_results = state.get('safety_results') or {}
            deficiency_results = state.get('deficiency_results') or {}
            recommendation_results = state.get('recommendation_results') or {}

            print(f"   ✓ Complete")

            return {
                'test_id': test_id,
                'question': question,
                'description': test_case['description'],
                'test_type': test_case['test_type'],
                'patient_profile': profile,
                # Pipeline metadata
                'iterations': state.get('iterations', 0),
                'evidence_chain': state.get('evidence_chain', []),
                # Entity extraction
                'medications_list': state.get('medications_list', []),
                'supplements_list': state.get('supplements_list', []),
                'conditions_list': state.get('conditions_list', []),
                'dietary_restrictions_list': state.get('dietary_restrictions_list', []),
                'candidate_supplements_list': state.get('candidate_supplements_list', []),
                # Specialist flags
                'safety_checked': state.get('safety_checked', False),
                'deficiency_checked': state.get('deficiency_checked', False),
                'recommendations_checked': state.get('recommendations_checked', False),
                # Specialist results
                'safety_results': safety_results,
                'safety_interactions': safety_results.get('interactions', []),
                'deficiency_results': deficiency_results,
                'all_at_risk': deficiency_results.get('all_at_risk', []),
                'critical_overlaps': deficiency_results.get('critical_overlaps', []),
                'recommendation_results': recommendation_results,
                'candidates': recommendation_results.get('recommendations', []),
                # Final output
                'final_answer': actual_answer,
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                'test_id': test_id,
                'question': question,
                'description': test_case['description'],
                'test_type': test_case['test_type'],
                'patient_profile': profile,
                'iterations': 0,
                'evidence_chain': [],
                'medications_list': [],
                'supplements_list': [],
                'conditions_list': [],
                'dietary_restrictions_list': [],
                'candidate_supplements_list': [],
                'safety_checked': False,
                'deficiency_checked': False,
                'recommendations_checked': False,
                'safety_results': {},
                'safety_interactions': [],
                'deficiency_results': {},
                'all_at_risk': [],
                'critical_overlaps': [],
                'recommendation_results': {},
                'candidates': [],
                'final_answer': f"ERROR: {str(e)}",
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases."""
        if not self.test_cases:
            self.load_test_cases()

        print(f"🚀 Running {len(self.test_cases)} multi-agent test cases...\n")
        print("=" * 80)

        results = []
        for test_case in self.test_cases:
            result = self.run_test_case(test_case)
            results.append(result)
            print()

        self.results = results
        return results

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a human-reviewable text report.

        Structure per test:
          - Question + patient profile
          - Normalized entities
          - Specialists that ran + iteration count
          - Evidence chain (supervisor routing log)
          - Safety interactions (if any)
          - Deficiency findings (if any)
          - Recommendation candidates (if any)
          - Final answer
        """
        if not self.results:
            return "No test results available. Run tests first."

        total_tests = len(self.results)

        # ── Header ──
        report_lines = [
            "=" * 80,
            "TEST REPORT — Multi-Agent Tests (Human Review)",
            "=" * 80,
            f"Dataset: {self.dataset_name}",
            f"Total Tests: {total_tests}",
            "",
        ]

        # ── One block per test ──
        report_lines += [
            "=" * 80,
            "TEST RESULTS",
            "=" * 80,
            "",
        ]

        for result in self.results:
            profile = result.get('patient_profile', {})

            report_lines += [
                f"Test {result['test_id']}",
                "-" * 60,
                f"  Question:  {result['question']}",
                f"  Type:      {result.get('test_type', 'unknown')}",
                f"  Desc:      {result.get('description', '')}",
                "",
            ]

            # Patient profile
            report_lines += [
                "  PATIENT PROFILE:",
                f"    Medications:          {', '.join(profile.get('medications', [])) or 'None'}",
                f"    Supplements:          {', '.join(profile.get('supplements', [])) or 'None'}",
                f"    Conditions:           {', '.join(profile.get('conditions', [])) or 'None'}",
                f"    Dietary Restrictions: {', '.join(profile.get('dietary_restrictions', [])) or 'None'}",
                "",
            ]

            # Normalized entities
            report_lines += [
                "  NORMALIZED ENTITIES:",
                f"    Medications:          {result.get('medications_list', []) or 'None'}",
                f"    Supplements:          {result.get('supplements_list', []) or 'None'}",
                f"    Conditions:           {result.get('conditions_list', []) or 'None'}",
                f"    Dietary Restrictions: {result.get('dietary_restrictions_list', []) or 'None'}",
                f"    Candidate Supplements: {result.get('candidate_supplements_list', []) or 'None'}",
                "",
            ]

            # Pipeline metadata
            specialists_ran = []
            if result.get('safety_checked'):
                specialists_ran.append('Safety')
            if result.get('deficiency_checked'):
                specialists_ran.append('Deficiency')
            if result.get('recommendations_checked'):
                specialists_ran.append('Recommendation')

            report_lines += [
                "  PIPELINE METADATA:",
                f"    Iterations:      {result.get('iterations', 0)}",
                f"    Specialists ran: {', '.join(specialists_ran) or 'None'}",
                "",
            ]

            # Evidence chain (supervisor routing log)
            evidence_chain = result.get('evidence_chain', [])
            if evidence_chain:
                report_lines.append("  EVIDENCE CHAIN:")
                for i, step in enumerate(evidence_chain, 1):
                    report_lines.append(f"    {i}. {step}")
                report_lines.append("")
            else:
                report_lines += ["  EVIDENCE CHAIN: Empty", ""]

            # Safety interactions
            safety_interactions = result.get('safety_interactions', [])
            if safety_interactions:
                report_lines.append("  SAFETY INTERACTIONS:")
                for ix in safety_interactions:
                    supplement = ix.get('supplement', '?')
                    target = ix.get('target', '?')
                    pathway = ix.get('pathway', '?')
                    severity = ix.get('severity', '?')
                    description = ix.get('description', '')
                    report_lines.append(
                        f"    [{severity}] {supplement} ↔ {target} | Pathway: {pathway}"
                    )
                    if description:
                        report_lines.append(f"      Description: {description}")
                report_lines.append("")
            else:
                safety_status = result.get('safety_results', {}).get('status', 'did not run')
                report_lines += [f"  SAFETY INTERACTIONS: None (status: {safety_status})", ""]

            # Deficiency findings
            all_at_risk = result.get('all_at_risk', [])
            critical_overlaps = result.get('critical_overlaps', [])
            if all_at_risk:
                report_lines.append("  DEFICIENCY FINDINGS:")
                report_lines.append(f"    Nutrients at risk: {', '.join(all_at_risk)}")
                if critical_overlaps:
                    report_lines.append("    Critical overlaps:")
                    for ov in critical_overlaps:
                        report_lines.append(
                            f"      🚨 {ov.get('nutrient', '?')}: {ov.get('warning', '')}"
                        )
                report_lines.append("")
            else:
                def_status = result.get('deficiency_results', {}).get('status', 'did not run')
                report_lines += [f"  DEFICIENCY FINDINGS: None (status: {def_status})", ""]

            # Recommendation candidates
            candidates = result.get('candidates', [])
            if candidates:
                report_lines.append("  RECOMMENDATION CANDIDATES:")
                for c in candidates:
                    name = c.get('supplement_name', '?')
                    rating = c.get('safety_rating', '?')
                    treated = c.get('symptom_treated', '?')
                    report_lines.append(
                        f"    {name} — safety_rating: {rating} | treats: {treated}"
                    )
                report_lines.append("")
            else:
                rec_status = result.get('recommendation_results', {}).get('status', 'did not run')
                report_lines += [f"  RECOMMENDATION CANDIDATES: None (status: {rec_status})", ""]

            # Final answer
            report_lines.append("  FINAL ANSWER:")
            final_answer = result.get('final_answer', '')
            for line in final_answer.strip().splitlines():
                report_lines.append(f"    {line}")
            report_lines += ["", ""]

        report = "\n".join(report_lines)

        # ── Save report ──
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            reports_dir = self.tests_dir / "reports"
            reports_dir.mkdir(exist_ok=True)

            report_file = reports_dir / f"report_{timestamp}.txt"

            # Handle same-second collision
            if report_file.exists():
                suffix = 1
                while (reports_dir / f"report_{timestamp}_{suffix}.txt").exists():
                    suffix += 1
                report_file = reports_dir / f"report_{timestamp}_{suffix}.txt"

            output_path = report_file
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 Report saved to {output_path}")

        return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run multi-agent tests against golden dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='golden_dataset_multi_agent',
        help='Name of the dataset without .csv extension (default: golden_dataset_multi_agent)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Direct path to CSV file (overrides --dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save report (default: tests/multi_agent/reports/report_{timestamp}.txt)'
    )
    parser.add_argument(
        '--test-id',
        type=str,
        default=None,
        help='Run only a specific test ID'
    )

    args = parser.parse_args()

    if args.dataset_path:
        runner = MultiAgentTestRunner(golden_dataset_path=args.dataset_path)
    else:
        runner = MultiAgentTestRunner(dataset_name=args.dataset)

    runner.load_test_cases()

    if args.test_id:
        test_case = next((tc for tc in runner.test_cases if tc['test_id'] == args.test_id), None)
        if not test_case:
            print(f"❌ Test ID {args.test_id} not found")
            return
        result = runner.run_test_case(test_case)
        runner.results = [result]
    else:
        runner.run_all_tests()

    report = runner.generate_report(output_path=args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
