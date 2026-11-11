"""
Test Runner — Safety Tests (Human-in-the-Loop)

Loads golden_dataset_safety.csv, runs the workflow for each test case,
and generates a plain text report for human review.

No automated scoring or keyword matching — output is reviewed manually.

Report saved to:
  tests/safety/reports/test_report_safety_{timestamp}/report.txt
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


class SafetyTestRunner:
    """Runs safety tests and generates a human-reviewable report."""

    def __init__(self, dataset_name: str = None, golden_dataset_path: str = None):
        """
        Initialize test runner.

        Args:
            dataset_name: Name of the dataset file without .csv extension.
                          Looks in the same folder as this script.
            golden_dataset_path: Direct path to CSV file (overrides dataset_name).
        """
        self.tests_dir = Path(__file__).parent
        self.test_type = "safety"

        if golden_dataset_path:
            self.golden_dataset_path = Path(golden_dataset_path)
        elif dataset_name:
            self.golden_dataset_path = self.tests_dir / f"{dataset_name}.csv"
        else:
            self.golden_dataset_path = self.tests_dir / "golden_dataset_safety.csv"

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
        Run a single test case through the workflow.

        Captures:
          - generated_safety_queries (cypher, executed, error per supplement)
          - safety_results.interactions
          - final_answer
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

            generated_safety_queries = state.get('generated_safety_queries', [])
            safety_interactions = (
                (state.get('safety_results') or {}).get('interactions', [])
                )

            print(f"   ✓ Complete")

            return {
                'test_id': test_id,
                'question': question,
                'description': test_case['description'],
                'test_type': test_case['test_type'],
                'patient_profile': profile,
                'generated_safety_queries': generated_safety_queries,
                'safety_interactions': safety_interactions,
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
                'generated_safety_queries': [],
                'safety_interactions': [],
                'final_answer': f"ERROR: {str(e)}",
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases."""
        if not self.test_cases:
            self.load_test_cases()

        print(f"🚀 Running {len(self.test_cases)} safety test cases...\n")
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
          - Safety queries (per supplement: cypher, executed, error)
          - Interactions found
          - Final answer
        """
        if not self.results:
            return "No test results available. Run tests first."

        total_tests = len(self.results)

        # ── Header ──
        report_lines = [
            "=" * 80,
            "TEST REPORT — Safety Tests (Human Review)",
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

            # Safety queries — one block per supplement
            safety_queries = result.get('generated_safety_queries', [])
            if safety_queries:
                report_lines.append("  SAFETY QUERIES:")
                for entry in safety_queries:
                    supplement = entry.get('supplement', 'Unknown')
                    executed = entry.get('executed', False)
                    error = entry.get('error', None)
                    cypher = entry.get('cypher', '')

                    report_lines += [
                        f"    Supplement: {supplement}",
                        f"      Executed: {executed}",
                        f"      Error:    {error if error else 'None'}",
                        f"      Cypher:",
                    ]
                    if cypher:
                        for line in cypher.strip().splitlines():
                            report_lines.append(f"        {line}")
                    else:
                        report_lines.append("        (no query generated)")
                    report_lines.append("")
            else:
                report_lines += ["  SAFETY QUERIES: None", ""]

            # Interactions found
            interactions = result.get('safety_interactions', [])
            if interactions:
                report_lines.append("  INTERACTIONS FOUND:")
                for ix in interactions:
                    supplement = ix.get('supplement', '?')
                    target = ix.get('target', '?')
                    pathway = ix.get('pathway', '?')
                    severity = ix.get('severity', '?')
                    description = ix.get('description', '')
                    detail = ix.get('detail', '')
                    report_lines.append(
                        f"    [{severity}] {supplement} ↔ {target} | Pathway: {pathway}"
                    )
                    if description:
                        report_lines.append(f"      Description: {description}")
                    if detail:
                        report_lines.append(f"      Detail:      {detail}")
                report_lines.append("")
            else:
                report_lines += ["  INTERACTIONS FOUND: None", ""]

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

    parser = argparse.ArgumentParser(description='Run safety tests against golden dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='golden_dataset_safety',
        help='Name of the dataset without .csv extension (default: golden_dataset_safety)'
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
        help='Path to save report (default: tests/safety/reports/test_report_safety_{timestamp}/report.txt)'
    )
    parser.add_argument(
        '--test-id',
        type=str,
        default=None,
        help='Run only a specific test ID'
    )

    args = parser.parse_args()

    if args.dataset_path:
        runner = SafetyTestRunner(golden_dataset_path=args.dataset_path)
    else:
        runner = SafetyTestRunner(dataset_name=args.dataset)

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