#!/bin/sh
begin=504376
end=504457
for i in $(seq ${begin} ${end})
do
    jdel $i
done