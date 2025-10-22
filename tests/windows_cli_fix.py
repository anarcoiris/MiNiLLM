#!/usr/bin/env python3
"""
fix_windows_cli.py

Parche automático para arreglar el problema de pipes en Windows
en interactive_cli.py

Problema: El separador <|doc|> causa error en cmd.exe porque | es un operador de pipe
Solución: Usar lista de argumentos en lugar de string, y escapar correctamente

Uso:
    python fix_windows_cli.py
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime


def backup_file(filepath: Path):
    """Crea backup del archivo"""
    backup_path = filepath.with_suffix(filepath.suffix + '.backup')
    counter = 1
    while backup_path.exists():
        backup_path = filepath.with_suffix(f'{filepath.suffix}.backup{counter}')
        counter += 1
    
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup creado: {backup_path}")
    return backup_path


def fix_run_command():
    """
    Arregla la función run_command en interactive_cli.py
    para que funcione correctamente en Windows con caracteres especiales
    """
    import re  # Import necesario para este fix
    
    cli_file = Path('interactive_cli.py')
    
    if not cli_file.exists():
        print("❌ Error: interactive_cli.py no encontrado")
        print("   Ejecuta este script desde el directorio del proyecto")
        return False
    
    print("📝 Leyendo interactive_cli.py...")
    content = cli_file.read_text(encoding='utf-8')
    
    # Verifica si ya está parcheado
    if 'WINDOWS_FIX_APPLIED' in content:
        print("✅ El parche ya está aplicado")
        return True
    
    # Crea backup
    backup_path = backup_file(cli_file)
    
    # Define la nueva función run_command
    new_run_command = '''def run_command(cmd: List[str], description: str = "Ejecutando comando") -> bool:
    """
    Ejecuta un comando y muestra progreso.
    
    WINDOWS_FIX_APPLIED: Manejo especial de caracteres especiales en Windows
    
    Returns:
        True si exitoso, False si falló
    """
    console.print(f"\\n[bold cyan]🚀 {description}...[/]")
    
    # En Windows, no mostrar el comando raw porque puede tener caracteres especiales
    if sys.platform == 'win32':
        console.print(f"[dim]Ejecutando comando (ver arriba para detalles)[/]\\n")
    else:
        console.print(f"[dim]Comando: {' '.join(cmd)}[/]\\n")
    
    try:
        # CRÍTICO: En Windows, NO usar shell=True con string
        # Siempre pasar lista de argumentos
        if sys.platform == 'win32':
            # Windows: usa lista directamente, subprocess.run maneja el escaping
            # NO unir en string ni usar shell=True
            result = subprocess.run(
                cmd,  # Lista, NO string
                check=True,
                capture_output=False,
                text=True,
                # NO shell=True aquí - causa problemas con |, &, etc
            )
        else:
            # Unix: también usa lista (más seguro)
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
        
        console.print(f"\\n[bold green]✅ {description} completado[/]\\n")
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"\\n[bold red]❌ Error en {description}[/]")
        console.print(f"[red]{e}[/]\\n")
        return False
    except FileNotFoundError:
        console.print(f"\\n[bold red]❌ Python no encontrado en el PATH[/]")
        console.print(f"[yellow]Verifica tu instalación de Python[/]\\n")
        return False
    except Exception as e:
        console.print(f"\\n[bold red]❌ Error inesperado: {e}[/]\\n")
        return False'''
    
    # Encuentra y reemplaza la función run_command original
    import re
    
    # Patrón para encontrar la función run_command
    pattern = r'def run_command\(.*?\).*?(?=\ndef\s|\nclass\s|\Z)'
    
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("❌ No se pudo encontrar la función run_command")
        print("   El archivo puede haber sido modificado")
        return False
    
    # Reemplaza
    new_content = content[:match.start()] + new_run_command + content[match.end():]
    
    # Guarda el archivo parcheado
    cli_file.write_text(new_content, encoding='utf-8')
    
    print("✅ Parche aplicado exitosamente")
    print(f"   Backup guardado en: {backup_path}")
    print(f"   Archivo parcheado: {cli_file}")
    
    return True


def add_windows_warning():
    """
    Añade un warning al inicio de main_menu en interactive_cli.py
    """
    
    cli_file = Path('interactive_cli.py')
    content = cli_file.read_text(encoding='utf-8')
    
    # Busca la función main_menu
    pattern = r'(def main_menu\(state: PipelineState\):.*?""".*?""")'
    
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("⚠️  No se pudo añadir warning en main_menu")
        return
    
    warning_code = '''
    
    # Windows: Aviso sobre caracteres especiales
    if sys.platform == 'win32':
        console.print("[yellow]ℹ️  Windows detectado: usando modo compatible[/]")
        console.print("[dim]   Los comandos se ejecutarán con escaping automático[/]\\n")
'''
    
    insert_pos = match.end()
    new_content = content[:insert_pos] + warning_code + content[insert_pos:]
    
    cli_file.write_text(new_content, encoding='utf-8')
    print("✅ Warning de Windows añadido")


def verify_fix():
    """Verifica que el fix se aplicó correctamente"""
    
    cli_file = Path('interactive_cli.py')
    
    if not cli_file.exists():
        return False
    
    content = cli_file.read_text(encoding='utf-8')
    
    # Verifica que tenga el marker
    if 'WINDOWS_FIX_APPLIED' not in content:
        print("❌ El parche no se aplicó correctamente")
        return False
    
    # Verifica que NO use shell=True en Windows
    if 'shell=True' in content and 'win32' in content:
        # Busca si está en la sección problemática
        lines = content.split('\n')
        in_run_command = False
        has_shell_true_bug = False
        
        for line in lines:
            if 'def run_command' in line:
                in_run_command = True
            elif in_run_command and 'def ' in line:
                in_run_command = False
            
            if in_run_command and 'shell=True' in line and 'win32' in content[max(0, content.find(line)-500):content.find(line)+500]:
                has_shell_true_bug = True
                break
        
        if has_shell_true_bug:
            print("⚠️  Advertencia: Todavía hay shell=True en código de Windows")
            return False
    
    print("✅ Fix verificado correctamente")
    return True


def show_test_command():
    """Muestra un comando de test"""
    print("\n" + "="*70)
    print("🧪 TESTING")
    print("="*70)
    print("\nPara verificar que el fix funciona, prueba:")
    print("\n1. Ejecuta el smoke test:")
    print("   python smoke_test.py")
    print("\n2. Si todo pasa, prueba el CLI interactivo:")
    print("   python interactive_cli.py")
    print("\n3. O prueba directamente con main.py:")
    print('   python main.py train --tokenizer tokenizer.json --corpus corpus.txt --doc-separator "<|doc|>" --outdir test')
    print("\n   (Nota: si tienes los archivos necesarios)")
    print("="*70)


def main():
    print("="*70)
    print("🔧 FIX PARA WINDOWS - Interactive CLI")
    print("="*70)
    print("\nProblema: El carácter | en <|doc|> causa error en cmd.exe")
    print("Solución: Usar subprocess con lista en lugar de string")
    print()
    
    if sys.platform != 'win32':
        print("⚠️  Este fix es para Windows.")
        print("   En tu sistema no debería ser necesario, pero se aplicará de todos modos.")
        print()
    
    # Aplica el fix
    if fix_run_command():
        # Añade warning
        try:
            import re
            add_windows_warning()
        except Exception as e:
            print(f"⚠️  No se pudo añadir warning (no crítico): {e}")
        
        # Verifica
        if verify_fix():
            print("\n" + "="*70)
            print("✅ FIX APLICADO EXITOSAMENTE")
            print("="*70)
            show_test_command()
            return 0
        else:
            print("\n❌ Hubo problemas al verificar el fix")
            return 1
    else:
        print("\n❌ No se pudo aplicar el fix")
        print("\nRevisa:")
        print("  1. Que estés en el directorio correcto")
        print("  2. Que interactive_cli.py exista")
        print("  3. Que tengas permisos de escritura")
        return 1


if __name__ == '__main__':
    sys.exit(main())