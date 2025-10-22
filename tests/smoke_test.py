#!/usr/bin/env python3
"""
smoke_test.py - Test Global de Mini-LLM v2

Verifica:
1. Versión de Python
2. Dependencias instaladas
3. PyTorch + CUDA
4. Módulos principales funcionan
5. CLI básico funciona
6. Tests de integración mínimos

Uso:
    python smoke_test.py
    python smoke_test.py --verbose
    python smoke_test.py --fix  # Intenta reparar problemas comunes
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import Tuple, Optional
import traceback

# Colores para terminal (compatible con Windows)
try:
    import colorama
    colorama.init()
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
except ImportError:
    GREEN = YELLOW = RED = BLUE = RESET = BOLD = ''


class TestResult:
    """Resultado de un test"""
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __repr__(self):
        status = f"{GREEN}✅ PASS{RESET}" if self.passed else f"{RED}❌ FAIL{RESET}"
        return f"{status} {self.name}: {self.message}"


class SmokeTest:
    """Sistema de tests completo"""
    
    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.results = []
        self.critical_failures = []
    
    def log(self, msg: str, level: str = "INFO"):
        """Log con colores"""
        if level == "INFO":
            color = BLUE
        elif level == "SUCCESS":
            color = GREEN
        elif level == "WARNING":
            color = YELLOW
        elif level == "ERROR":
            color = RED
        else:
            color = RESET
        
        print(f"{color}{msg}{RESET}")
    
    def log_verbose(self, msg: str):
        """Log solo en modo verbose"""
        if self.verbose:
            print(f"  {msg}")
    
    def add_result(self, result: TestResult):
        """Añade resultado de test"""
        self.results.append(result)
        if not result.passed:
            self.log(str(result), "ERROR")
            if result.details and self.verbose:
                print(f"    Details: {result.details}")
        else:
            self.log(str(result), "SUCCESS")
    
    # ==================== FASE 1: Verificación de Sistema ====================
    
    def test_python_version(self) -> TestResult:
        """Verifica versión de Python"""
        self.log("\n🐍 Verificando Python...", "INFO")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            return TestResult(
                "Python Version",
                False,
                f"Se requiere Python 3.10+, tienes {version_str}",
                "Descarga Python 3.10+ de python.org"
            )
        
        self.log_verbose(f"Python {version_str} en {sys.executable}")
        return TestResult("Python Version", True, f"Python {version_str} OK")
    
    def test_windows_specifics(self) -> TestResult:
        """Verifica configuración específica de Windows"""
        if sys.platform != 'win32':
            return TestResult("Windows Config", True, "No Windows, skip")
        
        self.log("\n🪟 Verificando configuración Windows...", "INFO")
        
        issues = []
        
        # Verifica UTF-8 mode
        if sys.getdefaultencoding() != 'utf-8':
            issues.append(f"Encoding por defecto es {sys.getdefaultencoding()}, no UTF-8")
        
        # Verifica long path support
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                r"SYSTEM\CurrentControlSet\Control\FileSystem")
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            if value != 1:
                issues.append("Long paths no habilitado (puede causar problemas)")
        except:
            pass
        
        if issues:
            return TestResult(
                "Windows Config",
                False,
                "; ".join(issues),
                "Considera habilitar UTF-8 mode y long paths"
            )
        
        return TestResult("Windows Config", True, "Configuración Windows OK")
    
    # ==================== FASE 2: Dependencias ====================
    
    def check_package(self, package_name: str, import_name: Optional[str] = None) -> TestResult:
        """Verifica si un paquete está instalado"""
        import_name = import_name or package_name
        spec = importlib.util.find_spec(import_name)
        
        if spec is None:
            return TestResult(
                f"Package: {package_name}",
                False,
                "No instalado",
                f"pip install {package_name}"
            )
        
        # Intenta obtener versión
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            self.log_verbose(f"{package_name} {version}")
        except:
            version = "unknown"
        
        return TestResult(
            f"Package: {package_name}",
            True,
            f"v{version}"
        )
    
    def test_pytorch_cuda(self) -> TestResult:
        """Verifica PyTorch y CUDA"""
        self.log("\n🔥 Verificando PyTorch + CUDA...", "INFO")
        
        try:
            import torch
            
            torch_version = torch.__version__
            self.log_verbose(f"PyTorch {torch_version}")
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                self.log_verbose(f"CUDA {cuda_version}")
                self.log_verbose(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Test simple de CUDA
                try:
                    x = torch.randn(10, 10).cuda()
                    y = x @ x.T
                    _ = y.cpu()
                    
                    return TestResult(
                        "PyTorch + CUDA",
                        True,
                        f"PyTorch {torch_version} + CUDA OK ({gpu_name})"
                    )
                except Exception as e:
                    return TestResult(
                        "PyTorch + CUDA",
                        False,
                        "CUDA detectado pero falla al ejecutar",
                        str(e)
                    )
            else:
                # Sin CUDA pero PyTorch instalado
                return TestResult(
                    "PyTorch + CUDA",
                    True,
                    f"PyTorch {torch_version} (solo CPU - será LENTO)",
                    "Considera instalar versión con CUDA para mejor rendimiento"
                )
        
        except ImportError:
            return TestResult(
                "PyTorch + CUDA",
                False,
                "PyTorch no instalado",
                "pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )
        except Exception as e:
            return TestResult(
                "PyTorch + CUDA",
                False,
                f"Error al verificar PyTorch: {str(e)}",
                traceback.format_exc()
            )
    
    def test_dependencies(self) -> list:
        """Verifica todas las dependencias"""
        self.log("\n📦 Verificando dependencias...", "INFO")
        
        packages = [
            ("tokenizers", None),
            ("rich", None),
            ("numpy", None),
        ]
        
        results = []
        for pkg, import_name in packages:
            results.append(self.check_package(pkg, import_name))
        
        return results
    
    # ==================== FASE 3: Módulos Principales ====================
    
    def test_module_imports(self) -> list:
        """Verifica que los módulos principales se puedan importar"""
        self.log("\n📚 Verificando módulos del proyecto...", "INFO")
        
        modules = [
            'model',
            'dataset', 
            'training',
            'generation',
        ]
        
        results = []
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                self.log_verbose(f"{module_name}.py importado OK")
                results.append(TestResult(
                    f"Module: {module_name}",
                    True,
                    "Import OK"
                ))
            except ImportError as e:
                results.append(TestResult(
                    f"Module: {module_name}",
                    False,
                    f"Import failed: {e}",
                    traceback.format_exc()
                ))
            except Exception as e:
                results.append(TestResult(
                    f"Module: {module_name}",
                    False,
                    f"Error al importar: {e}",
                    traceback.format_exc()
                ))
        
        return results
    
    def test_model_creation(self) -> TestResult:
        """Test de creación de modelo"""
        self.log("\n🤖 Verificando creación de modelo...", "INFO")
        
        try:
            from model import TinyGPTv2
            import torch
            
            # Crea modelo pequeño
            model = TinyGPTv2(
                vocab_size=1000,
                block_size=64,
                n_embd=128,
                n_layer=2,
                n_head=4
            )
            
            # Test forward pass
            x = torch.randint(0, 1000, (2, 32))
            with torch.no_grad():
                logits = model(x)
            
            assert logits.shape == (2, 32, 1000), f"Shape incorrecta: {logits.shape}"
            
            self.log_verbose(f"Modelo creado: {model.get_num_params():,} parámetros")
            
            return TestResult(
                "Model Creation",
                True,
                "Modelo funciona correctamente"
            )
        
        except AssertionError as e:
            return TestResult("Model Creation", False, f"Assertion failed: {e}")
        except Exception as e:
            return TestResult(
                "Model Creation",
                False,
                f"Error: {str(e)}",
                traceback.format_exc()
            )
    
    def test_tokenizer_basic(self) -> TestResult:
        """Test básico de tokenizer"""
        self.log("\n🔤 Verificando tokenizer...", "INFO")
        
        try:
            from tokenizers import Tokenizer, models, trainers
            
            # Crea tokenizer simple
            tokenizer = Tokenizer(models.BPE())
            
            self.log_verbose("Tokenizer BPE creado OK")
            
            return TestResult(
                "Tokenizer Basic",
                True,
                "Tokenizer funcionando"
            )
        except Exception as e:
            return TestResult(
                "Tokenizer Basic",
                False,
                f"Error: {str(e)}",
                traceback.format_exc()
            )
    
    # ==================== FASE 4: Tests de Integración ====================
    
    def test_cli_help(self) -> TestResult:
        """Verifica que main.py --help funcione"""
        self.log("\n⚙️ Verificando CLI básico...", "INFO")
        
        try:
            result = subprocess.run(
                [sys.executable, 'main.py', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return TestResult(
                    "CLI Help",
                    False,
                    f"Exit code: {result.returncode}",
                    result.stderr
                )
            
            if 'usage:' in result.stdout.lower():
                self.log_verbose("CLI help funciona OK")
                return TestResult("CLI Help", True, "CLI funcional")
            else:
                return TestResult(
                    "CLI Help",
                    False,
                    "Output inesperado",
                    result.stdout[:200]
                )
        
        except subprocess.TimeoutExpired:
            return TestResult(
                "CLI Help",
                False,
                "Timeout después de 10s"
            )
        except FileNotFoundError:
            return TestResult(
                "CLI Help",
                False,
                "main.py no encontrado",
                "Ejecuta este script desde el directorio del proyecto"
            )
        except Exception as e:
            return TestResult(
                "CLI Help",
                False,
                f"Error: {str(e)}",
                traceback.format_exc()
            )
    
    def test_dataset_basic(self) -> TestResult:
        """Test básico de dataset"""
        self.log("\n📊 Verificando dataset...", "INFO")
        
        try:
            from dataset import CausalTextDataset
            import torch
            
            # Crea dataset dummy
            dummy_ids = list(range(1000))
            dataset = CausalTextDataset(
                ids=dummy_ids,
                block_size=64,
                randomize=False
            )
            
            # Obtiene un sample
            x, y = dataset[0]
            
            assert isinstance(x, torch.Tensor), "x no es tensor"
            assert isinstance(y, torch.Tensor), "y no es tensor"
            assert x.shape[0] == 64, f"Shape incorrecta: {x.shape}"
            
            self.log_verbose(f"Dataset creado: {len(dataset)} samples")
            
            return TestResult(
                "Dataset Basic",
                True,
                "Dataset funciona correctamente"
            )
        
        except Exception as e:
            return TestResult(
                "Dataset Basic",
                False,
                f"Error: {str(e)}",
                traceback.format_exc()
            )
    
    # ==================== FASE 5: Tests de Windows ====================
    
    def test_windows_pipe_escaping(self) -> TestResult:
        """Verifica que los pipes se escapen correctamente en Windows"""
        if sys.platform != 'win32':
            return TestResult("Windows Pipe Escaping", True, "No Windows, skip")
        
        self.log("\n🪟 Verificando escape de caracteres especiales...", "INFO")
        
        try:
            # Test comando con pipe (el problema reportado)
            test_separator = "<|doc|>"
            
            # Intenta ejecutar comando con separator
            cmd = [
                sys.executable, '-c',
                f'import sys; print("{test_separator}")'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if test_separator in result.stdout:
                return TestResult(
                    "Windows Pipe Escaping",
                    True,
                    "Caracteres especiales manejados OK"
                )
            else:
                return TestResult(
                    "Windows Pipe Escaping",
                    False,
                    "Los pipes no se escapan correctamente",
                    "Esto causará problemas con --doc-separator"
                )
        
        except Exception as e:
            return TestResult(
                "Windows Pipe Escaping",
                False,
                f"Error: {str(e)}"
            )
    
    # ==================== FASE 6: Fixes Automáticos ====================
    
    def attempt_fix_torch(self):
        """Intenta reinstalar PyTorch"""
        self.log("\n🔧 Intentando reinstalar PyTorch...", "WARNING")
        
        try:
            # Desinstala versión actual
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'],
                capture_output=True
            )
            
            # Instala versión correcta para Windows
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'torch',
                 '--index-url', 'https://download.pytorch.org/whl/cu118'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("✅ PyTorch reinstalado", "SUCCESS")
                return True
            else:
                self.log(f"❌ Falló reinstalación: {result.stderr}", "ERROR")
                return False
        
        except Exception as e:
            self.log(f"❌ Error: {e}", "ERROR")
            return False
    
    def attempt_fix_dependencies(self):
        """Intenta instalar dependencias faltantes"""
        self.log("\n🔧 Instalando dependencias faltantes...", "WARNING")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("✅ Dependencias instaladas", "SUCCESS")
                return True
            else:
                self.log(f"❌ Falló instalación: {result.stderr}", "ERROR")
                return False
        
        except Exception as e:
            self.log(f"❌ Error: {e}", "ERROR")
            return False
    
    # ==================== Runner Principal ====================
    
    def run_all_tests(self):
        """Ejecuta todos los tests"""
        self.log(f"\n{BOLD}{'='*70}{RESET}")
        self.log(f"{BOLD}{BLUE}🔥 MINI-LLM v2 - SMOKE TEST{RESET}")
        self.log(f"{BOLD}{'='*70}{RESET}")
        
        # FASE 1: Sistema
        self.add_result(self.test_python_version())
        self.add_result(self.test_windows_specifics())
        
        # FASE 2: Dependencias
        pytorch_result = self.test_pytorch_cuda()
        self.add_result(pytorch_result)
        
        for result in self.test_dependencies():
            self.add_result(result)
        
        # Si PyTorch falló y --fix está activado
        if not pytorch_result.passed and self.fix:
            if self.attempt_fix_torch():
                # Re-test
                self.add_result(self.test_pytorch_cuda())
        
        # FASE 3: Módulos
        for result in self.test_module_imports():
            self.add_result(result)
        
        self.add_result(self.test_model_creation())
        self.add_result(self.test_tokenizer_basic())
        self.add_result(self.test_dataset_basic())
        
        # FASE 4: Integración
        self.add_result(self.test_cli_help())
        
        # FASE 5: Windows
        self.add_result(self.test_windows_pipe_escaping())
        
        # Resumen
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        self.log(f"\n{BOLD}{'='*70}{RESET}")
        self.log(f"{BOLD}📊 RESUMEN{RESET}")
        self.log(f"{BOLD}{'='*70}{RESET}")
        
        self.log(f"Tests ejecutados: {len(self.results)}")
        self.log(f"{GREEN}✅ Pasados: {passed}{RESET}")
        if failed > 0:
            self.log(f"{RED}❌ Fallidos: {failed}{RESET}")
        
        # Lista de tests fallidos
        if failed > 0:
            self.log(f"\n{RED}Tests fallidos:{RESET}")
            for r in self.results:
                if not r.passed:
                    self.log(f"  • {r.name}: {r.message}", "ERROR")
        
        # Recomendaciones
        if failed > 0:
            self.log(f"\n{YELLOW}💡 RECOMENDACIONES:{RESET}")
            
            # Chequea si PyTorch falló
            if any('pytorch' in r.name.lower() and not r.passed for r in self.results):
                self.log(f"  1. Reinstala PyTorch:")
                self.log(f"     pip install torch --index-url https://download.pytorch.org/whl/cu118")
            
            # Chequea si hay módulos que fallan
            if any('module' in r.name.lower() and not r.passed for r in self.results):
                self.log(f"  2. Verifica que estés en el directorio correcto")
                self.log(f"  3. Verifica que todos los archivos .py existan")
            
            # Chequea Windows pipe issue
            if any('pipe' in r.name.lower() and not r.passed for r in self.results):
                self.log(f"  4. El problema del pipe (|) en Windows necesita fix en interactive_cli.py")
        
        self.log(f"\n{BOLD}{'='*70}{RESET}")
        
        # Exit code
        return 0 if failed == 0 else 1


# ==================== Main ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smoke test para Mini-LLM v2')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output detallado')
    parser.add_argument('--fix', action='store_true', help='Intenta reparar problemas automáticamente')
    args = parser.parse_args()
    
    tester = SmokeTest(verbose=args.verbose, fix=args.fix)
    exit_code = tester.run_all_tests()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
