module tsvc_2_s275_mod
  use iso_c_binding
  implicit none
  integer, parameter :: B = 256
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name='tsvc_2_s275_fp64')
    ! Arguments: aa - inout, bb - in, cc - in, LEN_2D - length of one dimension
    real(c_double), intent(inout) :: aa(0:*)
    real(c_double), intent(in)    :: bb(0:*)
    real(c_double), intent(in)    :: cc(0:*)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t) :: ib, i, j, cur, idx, block_end, block_len
    real(c_double) :: prev(B)
    logical :: active(B)
    !$omp parallel do schedule(static) private(ib,i,j,cur,idx,block_end,block_len,prev,active) shared(aa,bb,cc,LEN_2D)
    do ib = 0_c_int64_t, LEN_2D - 1_c_int64_t, B
      block_end = min(ib + int(B, c_int64_t) - 1_c_int64_t, LEN_2D - 1_c_int64_t)
      block_len = block_end - ib + 1_c_int64_t
      ! Initialize condition and previous values for each column in the block
      do i = 0_c_int64_t, block_len - 1_c_int64_t
        idx = ib + i
        if (aa(idx) > 0.0d0) then
          active(i+1) = .true.
          prev(i+1) = aa(idx)
        else
          active(i+1) = .false.
        end if
      end do
      ! Process rows 1 .. LEN_2D-1
      do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
        !$omp simd
        do i = 0_c_int64_t, block_len - 1_c_int64_t
          if (.not. active(i+1)) cycle
          cur = j * LEN_2D + ib + i
          prev(i+1) = prev(i+1) + bb(cur) * cc(cur)
          aa(cur) = prev(i+1)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
