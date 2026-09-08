subroutine tsvc_2_s1232_fp64(a, b, c, LEN_2D, VLEN) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t), value, intent(in) :: VLEN
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: c(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, j_max

  ! Parallel region over rows (i). The inner loop over columns (j) has unit stride
  ! because we access a(j,i) where the first subscript varies fastest in Fortran's
  ! column-major storage. Use a dynamic schedule to balance the varying work per i.
  !$omp parallel shared(a,b,c,LEN_2D,VLEN) private(i,j,j_max)
  !$omp do schedule(dynamic)
  do i = 1, LEN_2D
    j_max = ((i - 1) / VLEN) + 1
    !$omp simd
    do j = 1, j_max
      a(j, i) = b(j, i) + c(j, i)
    end do
  end do
  !$omp end do
  !$omp end parallel

end subroutine tsvc_2_s1232_fp64
