DATASET="..datasets/anomaly_dataset.pickle"
TRAINDIR="../models/anomaly_new"

PERC=10
CORR=10

MODE=0

echo "=================================================="
echo "mislabel percentage: $PERC"
echo "=================================================="

if [ $MODE == 0 ]
    then
    echo "Train / Test mislabel model"
    python3 main.py \
    --data_path $DATASET \
    --train_dir "$TRAINDIR/mislabel-$PERC" \
    --train --mislabel --mislabel_perc $PERC --collect_train_loss --save_representer_dataset --test

    echo "Compute influences mislabel"
    python3 main.py \
    --data_path $DATASET \
    --train_dir "$TRAINDIR/mislabel-$PERC" \
    --mislabel --mislabel_perc $PERC --restore_best --influences --compute_score --compute_all_examples --compute_representer_influence

    echo "Compute each class influences mislabel"
    python3 main.py \
    --data_path $DATASET \
    --train_dir "$TRAINDIR/mislabel-$PERC" \
    --mislabel --mislabel_perc $PERC --restore_best --influences --compute_score --compute_all_examples --each_class
elif [ $MODE == 1 ]
    then
    for RUN in {0..9..1}
    do
        echo "Train / Test corrected model (loss based)"
        python3 main.py \
        --data_path $DATASET \
        --train_dir "$TRAINDIR/loss/$PERC-$CORR-$RUN" \
        --train --mislabel --mislabel_perc $PERC --manual_correction $CORR --correction_mode 1 --test \
        --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
        --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"
        
        echo "Train / Test corrected model (classwise absolute based)"
        python3 main.py \
        --data_path $DATASET \
        --train_dir "$TRAINDIR/classwise_absolute/$PERC-$CORR-$RUN" \
        --train --mislabel --mislabel_perc $PERC --manual_correction $CORR --correction_mode 0 --test \
        --absolute_influence --each_class \
        --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
        --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"

        echo "Train / Test corrected model (influence high based)"
        python3 main.py \
        --data_path $DATASET \
        --train_dir "$TRAINDIR/influence_high/$PERC-$CORR-$RUN" \
        --train --mislabel --mislabel_perc $PERC --manual_correction $CORR --correction_mode 0 --test \
        --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
        --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"
        
        echo "Train / Test representer based corrected model"
        python3 main.py \
        --data_path $DATASET \
        --train_dir "$TRAINDIR/representer_based/$PERC-$CORR-$RUN" \
        --train --mislabel --mislabel_perc $PERC --manual_correction $CORR --correction_mode 2 --test \
        --representer_path "$TRAINDIR/mislabel-$PERC/representer_data/"
    done
elif [ $MODE == 2 ]
    then 
        vals=( 10 25 50)
        for VAL in "${vals[@]}"
        do
            PERC=$VAL
            CORR=$VAL

            for RUN in {0..9..1}
            do
            echo "Train / Test corrected model (high loss removed)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/loss_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_high --remove_perc $CORR --correction_mode 1 --test \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"
               
            echo "Train / Test corrected model (influence negative removed)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/influence_low_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_low --remove_perc $CORR --correction_mode 0 --test \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"
            
            echo "Train / Test corrected model (representer based removed low model)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/representer_low_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_low --remove_perc $CORR --correction_mode 2 --test \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace" \
            --representer_path "$TRAINDIR/mislabel-$PERC/representer_data/"
            

            echo "Train / Test corrected model (influence negative removed)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/influence_high_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_high --remove_perc $CORR --correction_mode 0 --test \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"
            
            echo "Train / Test corrected model (influence negative removed)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/classwise_absolute_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_high --remove_perc $CORR --correction_mode 0 --test \
            --absolute_influence --each_class \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace"

            echo "Train / Test corrected model (representer based removed low model)"
            python3 main.py \
            --data_path $DATASET \
            --train_dir "$TRAINDIR/representer_high_removed/$PERC-$CORR-$RUN" \
            --train --mislabel --mislabel_perc $PERC --remove_high --remove_perc $CORR --correction_mode 2 --test \
            --loss_file "$TRAINDIR/mislabel-$PERC/loss.npy" \
            --influence_path "$TRAINDIR/mislabel-$PERC/influence-workspace" \
            --representer_path "$TRAINDIR/mislabel-$PERC/representer_data/"

            done
        done
else
    echo "Train / Test corrected model"
    python3 main.py \
    --data_path $DATASET \
    --train_dir "$TRAINDIR/mislabel-$PERC" \
    --predict --predict_set 0 --predict_start 0 --predict_end 100
fi