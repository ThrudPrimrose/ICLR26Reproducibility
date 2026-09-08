subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, n
  integer(c_int64_t) :: W, H
  integer(c_int64_t) :: nt_i_aa, nt_j_aa, it, jt, ib, ie, jb, je
  integer(c_int64_t) :: nt_i_bb, nt_j_bb

  n = LEN_2D
  W = 8
  H = 512

  nt_i_aa = (n - 8 + W - 1) / W
  nt_j_aa = (n - 8 + H - 1) / H
  nt_i_bb = (n - 8 + H - 1) / H
  nt_j_bb = (n - 8 + W - 1) / W

  !$omp parallel private(i, j, it, jt, ib, ie, jb, je)

  do jt = 1, nt_j_aa
    jb = (jt - 1) * H + 9
    je = min(jb + H - 1, n)
    !$omp do schedule(static)
    do it = 1, nt_i_aa
      ib = (it - 1) * W + 9
      ie = min(ib + W - 1, n)
      do j = jb, je
        !$omp simd
        do i = ib, ie
          aa(i, j) = aa(i, j - 1) + cc(i, j)
        end do
        !$omp end simd
      end do
    end do
    !$omp end do
  end do

  do it = 1, nt_i_bb
    ib = (it - 1) * H + 9
    ie = min(ib + H - 1, n)
    !$omp do schedule(static)
    do jt = 1, nt_j_bb
      jb = (jt - 1) * W + 9
      je = min(jb + W - 1, n)
      do i = ib, ie
        !$omp simd
        do j = jb, je
          bb(j, i) = bb(j, i - 1) + cc(j, i)
        end do
        !$omp end simd
      end do
    end do
    !$omp end do
  end do

  !$omp end parallel

end subroutine tsvc_2_s2233_fp64
