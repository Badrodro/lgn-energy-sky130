# DTN inference demo: classifies every test image stored in the weight
# memory with the DTN engine and checks the result against its label.
#
# Weight memory image (word addresses from 0x5000_0000, format in model/dtn/README.md):
#   0: magic 0x44544E31   1: number of layers L   2: number of images K
#   3: word address of the image table             4: N0 (input bytes per image)
#   8: L descriptors of 9 words: WBASE N R C NB REQ_M REQ_SH T0 KIND
#   image table: K x (ceil(N0/4) words of pixels + 1 word label)
#
# Ends with ebreak. Just before, x31 = 1 if every image is classified as its label,
# otherwise 0x100 + number of errors (0x2 if the image is missing).

.equ DTN_CTRL,   0x000
.equ DTN_STATUS, 0x004
.equ DTN_WBASE,  0x008
.equ DTN_RESULT, 0x028
.equ DTN_BUF0,   0x400

.text
_start:
    lui   s0, 0x50000            # s0 = weight memory
    lui   s1, 0x40000            # s1 = DTN engine
    lw    t0, 0(s0)
    li    t1, 0x44544E31         # magic "DTN1"
    bne   t0, t1, no_image
    lw    s2, 4(s0)              # L layers
    lw    s3, 8(s0)              # K images
    lw    s4, 12(s0)
    slli  s4, s4, 2
    add   s4, s4, s0             # s4 = byte address of the current image
    lw    s5, 16(s0)             # N0
    addi  s6, s5, 3
    srli  s6, s6, 2              # s6 = words per image
    addi  s10, s6, 1
    slli  s10, s10, 2            # s10 = image stride in bytes (pixels + label)
    li    s7, 0                  # errors
    li    s8, 0                  # image index

image_loop:
    bge   s8, s3, finished
    addi  t0, s1, DTN_BUF0       # copy the pixels into activation buffer 0
    mv    t1, s4
    mv    t2, s6
copy:
    lw    t3, 0(t1)
    sw    t3, 0(t0)
    addi  t1, t1, 4
    addi  t0, t0, 4
    addi  t2, t2, -1
    bnez  t2, copy
    lw    s9, 0(t1)              # label (word after the pixels)

    addi  t4, s0, 32             # descriptor table
    li    t5, 0                  # layer index
    li    t6, 0                  # input buffer of the layer
layer_loop:
    bge   t5, s2, layers_done
    addi  a2, s1, DTN_WBASE      # WBASE..T0: 8 words, same order as the registers
    mv    a3, t4
    li    a4, 8
regs:
    lw    a0, 0(a3)
    sw    a0, 0(a2)
    addi  a3, a3, 4
    addi  a2, a2, 4
    addi  a4, a4, -1
    bnez  a4, regs
    lw    a0, 32(t4)             # KIND
    slli  a0, a0, 1
    slli  a1, t6, 2
    or    a0, a0, a1
    ori   a0, a0, 1              # START
    sw    a0, DTN_CTRL(s1)
wait_done:
    lw    a0, DTN_STATUS(s1)
    andi  a0, a0, 2
    beqz  a0, wait_done
    xori  t6, t6, 1              # outputs become the next inputs
    addi  t4, t4, 36
    addi  t5, t5, 1
    j     layer_loop

layers_done:
    lw    a0, DTN_RESULT(s1)
    beq   a0, s9, correct
    addi  s7, s7, 1
correct:
    add   s4, s4, s10
    addi  s8, s8, 1
    j     image_loop

finished:
    bnez  s7, some_errors
    addi  x31, x0, 1
    ebreak
some_errors:
    addi  x31, s7, 0x100
    ebreak
no_image:
    addi  x31, x0, 2
    ebreak
