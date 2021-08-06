for folder in cmucovid coaid covid19fn covidcq recovery; do
    for file1 in $folder/*; do
        for file2 in $folder/*; do 
            if [ "$file1" != "$file2" ]; then
                for testtype in t-test; do
                    echo "Statistical test for $file1 and $file2 with test type $testtype \c"
                    python3 main.py $file1 $file2 0.05 $testtype
                done
            fi
        done
    done
done