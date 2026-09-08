subroutine wf_diff_skew_fp64(a, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: fi, fj
  real(c_double) :: t

  ! C: a[i,j] = a[i,j] + a[i-1,j] + a[i-1,j+1];  numpy (i,j) -> Fortran (j+1, i+1)
  ! Row chain is serial; fj is free + unit stride. One persistent team.
  ! K=2 pairing: in one barrier window compute rows fi+1 AND fi+2:
  !   t        = a(fj,fi)   + a(fj,fi-1) + a(fj+1,fi-1)   (== row fi+1, ref order)
  !   a(fj,fi+2) = t + a(fj,fi) + a(fj+1,fi)              (== row fi+2, ref order)
  ! bitwise identical to the reference (no reassociation); halves barrier count.
  if (len_2d < 2) return
  !$omp parallel
  do fi = 1, len_2d - 2, 2
    !$omp do simd
    do fj = 1, len_2d - 1
      t = a(fj, fi) + a(fj, fi - 1) + a(fj + 1, fi - 1)
      a(fj, fi + 1) = t
      a(fj, fi + 2) = t + a(fj, fi) + a(fj + 1, fi)
    end do
  end do
  if (mod(len_2d - 1, 2) == 1)
    !$omp do simd
    do fj = 1, len_2d - 1
      a(fj, len_2d) = a(fj, len_2d - 1) + a(fj, len_2d - 2) + a(fj + 1, len_2d - 2)
    end do
  endif
  !$omp end parallel
end subroutine wf_diff_skew_fp64
