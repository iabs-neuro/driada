#!/usr/bin/env python3
"""
Validate that all example scripts in the examples/ directory are runnable.

This script discovers and runs all Python scripts in the examples directory,
ensuring they execute without errors. It handles:
- Matplotlib backend configuration for headless execution
- Timeout management for long-running examples  
- Output capture and cleanup
- Detailed reporting of results

Usage:
    python tools/validate_examples.py [options]
    
Options:
    --quick        Run examples in quick mode where supported
    --timeout N    Set default timeout in seconds (default: 60)
    --pattern PAT  Only run examples matching pattern
    --verbose      Show script output
    --ci           CI mode - fail fast on first error
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# Configuration for specific examples
EXAMPLE_CONFIG = {
    # Long running examples need more time
    "intensive_analysis": {
        "circular_manifold/test_metrics.py": {"timeout": 300},
        "intense_dr_pipeline/intense_dr_pipeline.py": {"timeout": 180},
        "under_construction/selectivity_manifold_mapper/selectivity_manifold_mapper_demo.py": {"timeout": 300},
        "under_construction/selectivity_manifold_mapper/manifold_analysis_demo.py": {"timeout": 300},
    },
    # Examples that should be skipped in CI
    "skip_in_ci": [
        # Add examples that require special hardware or take too long
    ],
    # Examples that support --quick mode
    "supports_quick": [
        "dr_sequence/dr_sequence_neural_example.py",
        "under_construction/selectivity_manifold_mapper/selectivity_manifold_mapper_demo.py",
        "under_construction/selectivity_manifold_mapper/manifold_analysis_demo.py",
    ]
}


class ExampleValidator:
    """Validates example scripts by running them in controlled environment."""
    
    def __init__(self, quick_mode: bool = False, default_timeout: int = 60,
                 verbose: bool = False, ci_mode: bool = False):
        self.quick_mode = quick_mode
        self.default_timeout = default_timeout
        self.verbose = verbose
        self.ci_mode = ci_mode
        self.results = []
        
    def setup_environment(self) -> Dict[str, str]:
        """Setup environment for headless matplotlib execution."""
        env = os.environ.copy()
        # Force non-interactive backend
        env['MPLBACKEND'] = 'Agg'
        # Disable any GUI elements
        env['QT_QPA_PLATFORM'] = 'offscreen'
        return env
        
    def find_examples(self, pattern: Optional[str] = None) -> List[Path]:
        """Find all Python scripts in examples directory."""
        examples_dir = Path("examples")
        if not examples_dir.exists():
            raise FileNotFoundError("examples/ directory not found")
            
        scripts = []
        for script in examples_dir.rglob("*.py"):
            # Skip __pycache__ and test files
            if "__pycache__" in str(script) or script.name.startswith("test_"):
                continue
            # Apply pattern filter if provided
            if pattern and pattern not in str(script):
                continue
            scripts.append(script)
            
        return sorted(scripts)
        
    def get_example_config(self, script_path: Path) -> Dict:
        """Get configuration for specific example."""
        rel_path = str(script_path.relative_to("examples"))
        
        config = {"timeout": self.default_timeout, "args": []}
        
        # Check intensive examples
        if rel_path in EXAMPLE_CONFIG["intensive_analysis"]:
            config.update(EXAMPLE_CONFIG["intensive_analysis"][rel_path])
            
        # Add --quick flag if supported
        if self.quick_mode and rel_path in EXAMPLE_CONFIG["supports_quick"]:
            config["args"].append("--quick")
            
        # Check if should skip in CI
        if self.ci_mode and rel_path in EXAMPLE_CONFIG["skip_in_ci"]:
            config["skip"] = True
            config["skip_reason"] = "Skipped in CI mode"
            
        return config
        
    def create_run_script(self, example_path: Path, temp_dir: Path) -> Path:
        """Create a wrapper script that handles matplotlib backend."""
        wrapper_content = f'''
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

# Disable interactive mode
plt.ioff()

# Change to temp directory for output files
os.chdir(r"{temp_dir}")

# Add examples directory to path (some examples do this)
sys.path.insert(0, r"{example_path.parent}")

# Patch plt.show to be no-op
_original_show = plt.show
def _patched_show(*args, **kwargs):
    # Save current figure instead of showing
    if plt.get_fignums():
        for num in plt.get_fignums():
            fig = plt.figure(num)
            fig.savefig(f'figure_{{num}}.png', dpi=100, bbox_inches='tight')
        plt.close('all')
        
plt.show = _patched_show

# Now run the actual example
exec(open(r"{example_path}").read())
'''
        
        wrapper_path = temp_dir / "wrapper.py"
        wrapper_path.write_text(wrapper_content)
        return wrapper_path
        
    def run_example(self, script_path: Path) -> Tuple[bool, str, float]:
        """Run a single example script and return success status."""
        config = self.get_example_config(script_path)
        
        # Check if should skip
        if config.get("skip", False):
            return True, config.get("skip_reason", "Skipped"), 0.0
            
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create wrapper script
            wrapper_path = self.create_run_script(script_path, temp_path)
            
            # Build command
            cmd = [sys.executable, str(wrapper_path)] + config["args"]
            
            # Setup environment
            env = self.setup_environment()
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config["timeout"],
                    env=env,
                    cwd=str(script_path.parent)  # Run in script's directory
                )
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    # Check if any output files were created
                    output_files = list(temp_path.glob("*"))
                    msg = f"Success in {elapsed:.1f}s"
                    if output_files:
                        msg += f" ({len(output_files)} output files)"
                    return True, msg, elapsed
                else:
                    error_msg = result.stderr or result.stdout
                    # Truncate long errors
                    if len(error_msg) > 500:
                        error_msg = error_msg[:500] + "... (truncated)"
                    return False, f"Failed with: {error_msg}", elapsed
                    
            except subprocess.TimeoutExpired:
                return False, f"Timeout after {config['timeout']}s", config['timeout']
            except Exception as e:
                return False, f"Exception: {str(e)}", 0.0
                
    def validate_all(self, pattern: Optional[str] = None) -> bool:
        """Validate all example scripts."""
        scripts = self.find_examples(pattern)
        
        if not scripts:
            print("No example scripts found!")
            return False
            
        print(f"\nValidating {len(scripts)} example scripts...")
        print("=" * 80)
        
        all_success = True
        
        for i, script in enumerate(scripts, 1):
            rel_path = script.relative_to("examples")
            print(f"\n[{i}/{len(scripts)}] {rel_path}")
            print("-" * 40)
            
            success, message, elapsed = self.run_example(script)
            
            self.results.append({
                "script": str(rel_path),
                "success": success,
                "message": message,
                "elapsed": elapsed
            })
            
            if success:
                print(f"✓ {message}")
            else:
                print(f"✗ {message}")
                all_success = False
                
                if self.ci_mode:
                    print("\nCI mode: Failing fast on first error")
                    break
                    
            if self.verbose and not success:
                print(f"\nFull path: {script}")
                
        return all_success
        
    def print_summary(self):
        """Print summary of validation results."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"\nTotal scripts: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print("\nFailed scripts:")
            for r in failed:
                print(f"  - {r['script']}: {r['message']}")
                
        # Print timing statistics
        if successful:
            times = [r["elapsed"] for r in successful if r["elapsed"] > 0]
            if times:
                print(f"\nTiming statistics:")
                print(f"  Total time: {sum(times):.1f}s")
                print(f"  Average: {sum(times)/len(times):.1f}s")
                print(f"  Min: {min(times):.1f}s")
                print(f"  Max: {max(times):.1f}s")
                
    def save_report(self, output_path: str = "example_validation_report.json"):
        """Save detailed report to JSON file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "successful": len([r for r in self.results if r["success"]]),
                "failed": len([r for r in self.results if not r["success"]])
            },
            "config": {
                "quick_mode": self.quick_mode,
                "default_timeout": self.default_timeout
            },
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDetailed report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true',
                       help='Run examples in quick mode where supported')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Default timeout in seconds (default: 60)')
    parser.add_argument('--pattern', type=str,
                       help='Only run examples matching pattern')
    parser.add_argument('--verbose', action='store_true',
                       help='Show script output')
    parser.add_argument('--ci', action='store_true',
                       help='CI mode - fail fast on first error')
    parser.add_argument('--report', type=str, default='example_validation_report.json',
                       help='Output report file path')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create validator
    validator = ExampleValidator(
        quick_mode=args.quick,
        default_timeout=args.timeout,
        verbose=args.verbose,
        ci_mode=args.ci
    )
    
    # Run validation
    success = validator.validate_all(pattern=args.pattern)
    
    # Print summary
    validator.print_summary()
    
    # Save report
    validator.save_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()