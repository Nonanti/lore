Output demo.gif

Set Shell bash
Set FontSize 14
Set Width 1200
Set Height 400
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

Hide
Type "rm -rf /tmp/lore-demo3 && export LORE_DATA=/tmp/lore-demo3 LORE_LLM_BASE=http://localhost:11434/v1 LORE_LLM_MODEL=qwen3:8b PATH=$PWD/target/debug:$PATH && lore list >/dev/null 2>&1 && clear"
Enter
Show

Type "lore new-agent --name Aria --role 'research assistant' --traits curious,meticulous"
Sleep 400ms
Enter
Sleep 1200ms

Type "ID=$(lore list | awk 'NR==1{print $1}')"
Enter
Sleep 600ms

Type `lore remember $ID --title "Berkant's coffee" --body "Berkant drinks his coffee black, no sugar, always before 9am"`
Sleep 400ms
Enter
Sleep 1200ms

Type "lore ask $ID 'how does Berkant take his coffee?'"
Sleep 400ms
Enter
Hide
Sleep 16s
Show
Sleep 2500ms

Type "lore recall $ID coffee --semantic --limit 3"
Sleep 400ms
Enter
Sleep 4s
