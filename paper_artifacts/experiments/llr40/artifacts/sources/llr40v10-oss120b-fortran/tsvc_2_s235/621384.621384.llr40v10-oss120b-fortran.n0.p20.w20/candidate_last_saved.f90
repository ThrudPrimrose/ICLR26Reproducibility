module tsvc_2_s235_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
    ! Arguments match the C reference: a, aa, b, bb, c, LEN_2D
    real(c_double), dimension(*), intent(inout) :: a
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(in) :: b
    real(c_double), dimension(*), intent(in) :: bb
    real(c_double), dimension(*), intent(in) :: c
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j

    ! Update a: a[i] += b[i] * c[i]
    !$omp parallel do default(none) shared(a,b,c,LEN_2D) private(i)
    do i = 1, LEN_2D
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do

    ! Update aa using recurrence across the second dimension.
    ! aa(i,j) = aa(i,j-1) + bb(i,j) * a(i) for j = 2..LEN_2D
    !$omp parallel default(none) shared(aa,bb,a,LEN_2D) private(i,j)
    do j = 2, LEN_2D
      !$omp do schedule(static)
      do i = 1, LEN_2D
        aa(i + (j-1) * LEN_2D) = aa(i + (j-2) * LEN_2D) + bb(i + (j-1) * LEN_2D) * a(i)
      end do
      !$omp end do
    end do
    !$omp end parallel

  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235_mod
