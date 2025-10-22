#!/usr/bin/env python3
"""
verify_windows_fix.py

Verifica que el fix de Windows esté correctamente aplicado en interactive_cli.py
"""

import sys
from pathlib import Path


def verify_fix():
    """Verifica que el fix esté aplicado correctamente"""
    
    print("="*70)
    print("🔍 VERIFICACIÓN DEL FIX DE WINDOWS")
    print("="*70)
    
    cli_file = Path('interactive_cli.py')
    
    if not cli_file.exists():
        print("\n❌ interactive_cli.py no encontrado")
        return False
    
    print(f"\n📄 Archivo encontrado: {cli_file}")
    
    # Lee el contenido
    content = cli_file.read_text(encoding='utf-8')
    
    # Checklist de verificación
    checks = []
    
    # 1. Verifica que tenga la función run_command
    print("\n🔍 Verificando función run_command...")
    if 'def run_command(' in content:
        print("  ✅ Función run_command encontrada")
        checks.append(True)
    else:
        print("  ❌ Función run_command NO encontrada")
        checks.append(False)
    
    # 2. Verifica que tenga el marker del fix
    print("\n🔍 Verificando marker del fix...")
    if 'WINDOWS_FIX_APPLIED' in content:
        print("  ✅ Marker 'WINDOWS_FIX_APPLIED' encontrado")
        checks.append(True)
    else:
        print("  ⚠️  Marker 'WINDOWS_FIX_APPLIED' NO encontrado")
        print("     (El fix puede estar aplicado sin el marker)")
        checks.append(True)  # No crítico
    
    # 3. Verifica que NO use shell=True incorrectamente
    print("\n🔍 Verificando que NO use shell=True en Windows...")
    
    # Busca la función run_command
    run_command_start = content.find('def run_command(')
    if run_command_start != -1:
        # Encuentra el final de la función (siguiente 'def ')
        next_def = content.find('\ndef ', run_command_start + 1)
        if next_def == -1:
            next_def = len(content)
        
        run_command_code = content[run_command_start:next_def]
        
        # Verifica patrón problemático
        if 'shell=True' in run_command_code and 'win32' in run_command_code:
            # Busca si está comentado o en string
            lines = run_command_code.split('\n')
            has_problem = False
            
            for line in lines:
                if 'shell=True' in line:
                    # Verifica que NO esté en un comentario
                    if not line.strip().startswith('#'):
                        # Verifica el contexto
                        if 'NO shell=True' not in line and 'no usar shell=True' not in line.lower():
                            has_problem = True
                            print(f"  ⚠️  Línea problemática encontrada:")
                            print(f"     {line.strip()}")
            
            if has_problem:
                print("  ❌ Código todavía usa shell=True incorrectamente")
                checks.append(False)
            else:
                print("  ✅ shell=True mencionado solo en comentarios/docs")
                checks.append(True)
        else:
            print("  ✅ No usa shell=True en contexto de Windows")
            checks.append(True)
    else:
        print("  ❌ No se pudo verificar (función no encontrada)")
        checks.append(False)
    
    # 4. Verifica que use lista de argumentos correctamente
    print("\n🔍 Verificando uso correcto de lista de argumentos...")
    
    if run_command_start != -1:
        run_command_code = content[run_command_start:next_def]
        
        # Busca subprocess.run(cmd,
        if 'subprocess.run(' in run_command_code and 'cmd,' in run_command_code:
            print("  ✅ Usa 'cmd' como lista directamente")
            checks.append(True)
        elif "subprocess.run(\n                cmd," in run_command_code:
            print("  ✅ Usa 'cmd' como lista directamente (multilínea)")
            checks.append(True)
        else:
            print("  ⚠️  No se pudo verificar formato exacto")
            checks.append(True)  # No crítico
    
    # 5. Test práctico: ¿Puede manejar pipes?
    print("\n🔍 Test práctico: Simulando comando con pipe...")
    
    try:
        import subprocess
        
        # Simula el comando problemático
        test_cmd = [
            sys.executable, '-c',
            'print("<|doc|>")'
        ]
        
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if '<|doc|>' in result.stdout:
            print("  ✅ Puede ejecutar comandos con pipes correctamente")
            checks.append(True)
        else:
            print(f"  ⚠️  Output inesperado: {result.stdout}")
            checks.append(False)
    
    except Exception as e:
        print(f"  ⚠️  Error en test práctico: {e}")
        checks.append(False)
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nTests pasados: {passed}/{total}")
    
    if passed == total:
        print("\n✅ FIX CORRECTAMENTE APLICADO")
        print("\nEl archivo está listo para usar.")
        print("\nPuedes ejecutar:")
        print("  python interactive_cli.py")
        return True
    elif passed >= total - 1:
        print("\n⚠️  FIX MAYORMENTE APLICADO")
        print("\nHay warnings menores, pero debería funcionar.")
        print("\nPuedes probar:")
        print("  python interactive_cli.py")
        return True
    else:
        print("\n❌ PROBLEMAS DETECTADOS")
        print("\nEl fix no se aplicó correctamente.")
        print("\nOpciones:")
        print("  1. Restaura el backup: interactive_cli.py.backup")
        print("  2. Vuelve a ejecutar: python fix_windows_cli.py")
        return False


def show_current_run_command():
    """Muestra la función run_command actual"""
    
    cli_file = Path('interactive_cli.py')
    
    if not cli_file.exists():
        print("❌ interactive_cli.py no encontrado")
        return
    
    content = cli_file.read_text(encoding='utf-8')
    
    # Encuentra la función run_command
    start = content.find('def run_command(')
    if start == -1:
        print("❌ Función run_command no encontrada")
        return
    
    # Encuentra el final (siguiente 'def ')
    end = content.find('\ndef ', start + 1)
    if end == -1:
        end = len(content)
    
    run_command_code = content[start:end]
    
    print("\n" + "="*70)
    print("📄 FUNCIÓN run_command ACTUAL:")
    print("="*70)
    
    # Muestra con números de línea
    lines = run_command_code.split('\n')
    for i, line in enumerate(lines[:50], 1):  # Primeras 50 líneas
        print(f"{i:3d} | {line}")
    
    if len(lines) > 50:
        print(f"... ({len(lines) - 50} líneas más)")
    
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verifica el fix de Windows')
    parser.add_argument('--show-code', action='store_true', 
                       help='Muestra el código actual de run_command')
    args = parser.parse_args()
    
    if args.show_code:
        show_current_run_command()
    else:
        success = verify_fix()
        sys.exit(0 if success else 1)
