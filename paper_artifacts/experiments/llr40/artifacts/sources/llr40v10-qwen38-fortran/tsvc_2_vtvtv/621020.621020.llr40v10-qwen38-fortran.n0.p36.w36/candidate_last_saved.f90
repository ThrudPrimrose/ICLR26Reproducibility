subroutine tsvc_2_vtvtv_fp64(a, b, c, len1d) bind(C, name='tsvc_2_vtvtv_fp64')
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*)
  real(c_double), intent(in)    :: c(*)
  integer(c_int64_t), value     :: len1d
  integer(kind=8) :: i
  real(8) :: t1, tt
  integer :: j
  if (len1d <= 32768_8) then
    do j = 1, 3
      t1 = omp_get_wtime()
      do i = 1_8, len1d
        a(i) = a(i) * b(i) * c(i)
      end do
      tt = omp_get_wtime() - t1
      write(*,'(A,I1,A,F13.6)') 'ITER j=', j, ' us=', tt*1.0e6
      flush 6
    end do
  else
    !$omp parallel do default(none) shared(a,b,c,len1d)
    do i = 1_8, len1d
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do
  end if
end subroutine tsvc_2_vtvtv_fp64
