subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) &
    bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(c_double), dimension(*), intent(in) :: a
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), intent(in), value :: len_1d

  integer(c_int64_t) :: i, idx, first_pos, last_pos, cnt, neg_zero_cnt, pos_zero_cnt
  real(c_double) :: x, ai

  if (len_1d <= 0) then
     out_value = 0.0d0
     out_index = 0
     return
  end if

  x = a(1)
  idx = 0
  do i = 2, len_1d
     ai = a(i)
     if (ai > x) then
        x = ai
        idx = i - 1
     end if
  end do
  out_value = x
  out_index = idx

  write(6, '(A,I0)') 'N = ', len_1d
  write(6, '(A,ES25.16E3)') 'strict-scan max = ', x
  write(6, '(A,I0)') 'strict-scan idx = ', idx
  first_pos = -1
  last_pos = -1
  cnt = 0
  neg_zero_cnt = 0
  pos_zero_cnt = 0
  do i = 1, len_1d
     ai = a(i)
     if (ai == x) then
        if (first_pos < 0) first_pos = i - 1
        last_pos = i - 1
        cnt = cnt + 1
     end if
     if (ai == 0.0d0) then
        if (ieee_signbit(ai)) then
           neg_zero_cnt = neg_zero_cnt + 1
        else
           pos_zero_cnt = pos_zero_cnt + 1
        end if
     end if
  end do
  write(6, '(A,I0)') 'first tie pos = ', first_pos
  write(6, '(A,I0)') 'last tie pos = ', last_pos
  write(6, '(A,I0)') 'tie count = ', cnt
  write(6, '(A,I0)') 'neg zeros = ', neg_zero_cnt
  write(6, '(A,I0)') 'pos zeros = ', pos_zero_cnt
  if (x == 0.0d0) write(6, '(A,L1)') 'max has signbit = ', ieee_signbit(x)
  if (ieee_is_nan(x)) write(6, '(A)') 'MAX IS NAN'
  write(6, '(A)') '---- probe done ----'
  flush(6)
end subroutine argmax_with_index_fp64
