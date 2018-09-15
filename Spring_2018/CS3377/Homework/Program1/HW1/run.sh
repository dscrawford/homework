#Made by Daniel Crawford(Email dsc160130@utdallas.edu) on 1/20/2018
#cs3377.501
run_program() {
    echo Running 'hw1' with $# arguments:
    ./hw1 $@ >> stdout.log 2>> stderr.log
    echo -e "\tstdout appended to stdout.log"
    echo -e "\tstderr appended to stderr.log"
}
run_program
run_program abc
run_program a b c d e 
